import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from typing import List, Tuple, Union, Optional, Dict, Literal

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def _ensure_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    raise ValueError("El documento tiene que ser de tipo Dataframe")

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_categorical_series(s: pd.Series) -> bool:
    return (isinstance(s.dtype, CategoricalDtype) or
            pd.api.types.is_string_dtype(s) or
            pd.api.types.is_object_dtype(s))

class BaseTransform(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

# identificando las columnas de las variables numéricas y categóricas
@dataclass
class ColumnDetector(BaseTransform):
    numeric_columns: Optional[List[str]] = None
    categoric_columns: Optional[List[str]] = None

    numeric_detected_: Optional[List[str]] = None
    categoric_detected_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.numeric_columns is None:
            self.numeric_detected_ = [c for c in X.columns if _is_numeric_series(X[c])]
        else:
            self.numeric_detected_ = [c for c in self.numeric_columns if c in X.columns]

        if self.categoric_columns is None:
            self.categoric_detected_ = [c for c in X.columns if _is_categorical_series(X[c])]
        else:
            self.categoric_detected_ = [c for c in self.categoric_columns if c in X.columns]

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _ensure_dataframe(X)
        return X
    
# imputación de valores numéricos
@dataclass
class NumericImputer(BaseTransform):
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean"
    numeric_cols: Optional[List[str]] = None
    fill_value: Optional[float] = 0.0
    # imputer_num_: Optional[SimpleImputer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_cols is None:
            self.numeric_cols = [c for c in X.columns if _is_numeric_series(X[c])]

        X_num = X[self.numeric_cols]
        self.imputer_num_ = SimpleImputer(strategy=self.strategy,
                                         missing_values=np.nan).fit(X_num)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_num_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        
        X = _ensure_dataframe(X)
        X_num = X[self.numeric_cols]

        X_imputer = self.imputer_num_.transform(X_num)
        X.loc[:, self.numeric_cols] = X_imputer

        return X
    
@dataclass
class CategoricImputer(BaseTransform):
    strategy: Literal["most_frequent", "constant"] = "most_frequent"
    categoric_cols: Optional[List[str]] = None
    fill_value: Optional[str] = "other"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_cols is None:
            self.categoric_cols = [c for c in X.columns if _is_categorical_series(X[c])]
        
        X_cat = X[self.categoric_cols]
        self.imputer_cat_ = SimpleImputer(strategy=self.strategy,
                                          missing_values=np.nan).fit(X_cat)
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.imputer_cat_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        
        X = _ensure_dataframe(X)
        X_cat = X[self.categoric_cols]

        X_imputer = self.imputer_cat_.transform(X_cat)
        X.loc[:, self.categoric_cols] = X_imputer

        return X

# valores raros en las variables categóricas 
@dataclass
class RareCategoryGroup(BaseTransform):
    umbral: float = 0.04
    categoric_cols: Optional[List[str]] = None
    other_label: str = "other"
    keepers_: Dict[str, set] = field(default_factory=dict)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_cols is None:
            self.categoric_cols = [c for c in X.columns if _is_categorical_series(X[c])]
        
        self.keepers_ = {}
        for c in self.categoric_cols:
            frecuencia = X[c].value_counts(dropna=False) / len(X)
            keep = set(frecuencia[frecuencia >= self.umbral].index.to_list())
            self.keepers_[c] = keep
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keepers_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        
        for c, keep in self.keepers_.items():
            if c in X.columns:
                X[c] = X[c].astype("object").apply(lambda v: v if v in keep else self.other_label)

        return X
    
# manejo de outliers
@dataclass
class OutliersDetected(BaseTransform):
    mode: Literal["drop", "clip"] = "drop"
    limites_: Dict[str, tuple[float, float]] = field(default_factory = dict)
    numeric_cols: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_cols is None:
            self.numeric_cols = [c for c in X.columns if _is_categorical_series(X[c])]

        self.limites_ = {}
        for c in self.numeric_cols:
            q2 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q2

            low = q2 - 1.5*iqr
            high = q3 + 1.5*iqr
            self.limites_[c] = (low, high)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.limites_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        
        X = _ensure_dataframe(X)
        if self.mode == "clip":
            for c, (low, high) in self.limites_.items():
                if c in X.columns:
                    X[c] = X[c].clip(low, high)  
        if self.mode == "drop":
            for c, (low, high) in self.limites_.items():
                if c in X.columns:
                    X = X[(X[c] >= low) & (X[c] <= high)]
        else:
            raise ValueError("Solo se permiten 'drop' y 'clip'")
        
        return X

# escalamiento de valores numéricos    
@dataclass
class ScalerNum(BaseTransform):
    method: Literal["standard", "min_max"] = "standard"
    numeric_cols: List[str] = None
    scalers_: Optional[Union[StandardScaler, MinMaxScaler]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_cols is None:
            self.numeric_cols = [c for c in X.columns if _is_numeric_series(X[c])]

        X_num = X[self.numeric_cols]
        if self.method == "standard":
            self.scalers_ = StandardScaler().fit(X_num)
        elif self.method == "min_max":
            self.scalers_ = MinMaxScaler().fit(X_num)
        else:
            raise ValueError("Solo se permiten 'standard' y 'min_max'")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scalers_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        
        X = _ensure_dataframe(X)
        X_num = X[self.numeric_cols]
        X_sc = self.scalers_.transform(X_num)
        X.loc[:, self.numeric_cols] = X_sc

        return X
    
# escalamiento de valores categoricos
@dataclass
class ScalerCateg(BaseTransform):
    drop_first: Literal["first", "if_binary"] = "first"
    handle_unknown: Literal["ignore", "error"] = "ignore"
    sparse_output: Literal[True, False] = False
    categoric_cols: Optional[List[str]] = None
    # output_cols: Optional[List[str]] = None
    ohe_: Optional[OneHotEncoder] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_cols is None:
            self.categoric_cols = [c for c in X.columns if _is_categorical_series(X[c])]
        
        X_cat = X[self.categoric_cols]
        self.ohe_ = OneHotEncoder(drop=self.drop_first,
                                  handle_unknown=self.handle_unknown,
                                  sparse_output=self.sparse_output).fit(X_cat)
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ohe_ is None:
            raise RuntimeError("Primero se debe aplicar 'fit' antes de 'transform'")
        X = _ensure_dataframe(X)

        X_cat = X[self.categoric_cols]
        X_encoded = self.ohe_.transform(X_cat)
        features_names = self.ohe_.get_feature_names_out(self.categoric_cols).tolist()
        X = X.drop(self.categoric_cols, axis=1)
        X.loc[:, features_names] = X_encoded

        return X
    
# defieniendo el Pipeline
@dataclass
class Preprocessor(BaseTransform):
    numeric_impute_strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean"
    categorical_impute_strategy: Literal["most_frequent", "constant"] = "most_frequent"
    rare_umbral: float = 0.03
    outlier_mode: Literal["drop", "clip"] = "drop"
    scale_method: Literal["standard", "min_max"] = "standard"
    drop_first: Optional[str] = None

    numeric_cols: Optional[List[str]] = None
    categoric_cols: Optional[List[str]] = None
    pipeline_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)

        detector = ColumnDetector(numeric_columns=self.numeric_cols,
                                  categoric_columns=self.categoric_cols)
        detector.fit(X)
        self.numeric_cols = detector.numeric_detected_
        self.categoric_cols = detector.categoric_detected_

        self.pipeline_ = Pipeline(steps=[
            ("numeric_imputer", NumericImputer(strategy=self.categorical_impute_strategy,
                                               numeric_cols=self.numeric_cols)),
            ("categorical_imputer", CategoricImputer(strategy=self.categorical_impute_strategy,
                                                     categoric_cols=self.categoric_cols)),
            ("rare_grouper", RareCategoryGroup(umbral=self.rare_umbral,
                                               categoric_cols=self.categoric_cols)),
            ("outliers", OutliersDetected(mode=self.outlier_mode,
                                          numeric_cols=self.numeric_cols)),
            ("scaler_num", ScalerNum(method=self.scale_method,
                                     numeric_cols=self.numeric_cols)),
            ("scaler_cat", ScalerCateg(drop_first=self.drop_first,
                                       categoric_cols=self.categoric_cols))
        ])
        self.pipeline_.fit_transform(X)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Se debe llamar primero a 'fit' antes de 'transform'")
        
        X = _ensure_dataframe(X)
        X_processor = self.pipeline_.transform(X)

        return X_processor
    
if __name__ == "__main__":
    df = pd.DataFrame({"nombre": ["leo", "jose", "lui", "pablo", np.nan, "sofia", "sofia"],
                       "edad": [12, 14, 15, np.nan, 20, 23, np.nan]})
    
    procesor = Preprocessor()
    print(procesor.fit_transform(df))
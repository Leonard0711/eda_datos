import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from typing import List, Union, Optional, Dict, Literal, Sequence, Tuple

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

def _ensure_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    raise ValueError("El archivo tiene que ser de tipo DataFrame")

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_categorical_dtype(s: pd.Series) -> bool:
    return (isinstance(s.dtype, CategoricalDtype) or
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_string_dtype(s))

class BaseTransform(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass

    @abstractmethod
    def transform(sel, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

# deteccion de columnas numéricas y categóricas
@dataclass
class ColumnsDetector(BaseTransform):
    numeric_columns: Optional[List[str]] = None
    categoric_columns: Optional[List[str]] = None

    numeric_detected: Optional[List[str]] = None
    categoric_detected: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.numeric_columns is None:
            self.numeric_detected = [c for c in X.columns if _is_numeric_series(X[c])]
        else:
            self.numeric_detected = [c for c in self.numeric_columns if c in X.columns]

        if self.categoric_columns is None:
            self.categoric_detected = [c for c in X.columns if _is_categorical_dtype(X[c])]
        else:
            self.categoric_detected = [c for c in self.categoric_columns if _is_categorical_dtype(X[c])]

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _ensure_dataframe(X)
        return X

# detectando valores duplicados
@dataclass
class DetectedDuplicates(BaseTransform, BaseEstimator):
    keep: Literal["first", "last", False] = "first"
    subset: Optional[Union[str, Sequence[str]]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _ensure_dataframe(X)
        if isinstance(self.subset, (list, tuple)):
            subset_valid = [c for c in self.subset if c in X.columns]
            return X.drop_duplicates(keep=self.keep, subset=subset_valid or None)
        elif isinstance(self.subset, str):
            subset_valid = self.subset if self.subset in X.columns else None
            return X.drop_duplicates(keep=self.keep, subset=subset_valid)
        else:
            return X.drop_duplicates(keep=self.keep, subset=None)
        
# Imputacion de valores numéricas
@dataclass
class NumericImputer(BaseTransform, BaseEstimator):
    strategy: Literal["mean","median","most_frequent","constant"] = "mean"
    numeric_columns: Optional[List[str]] = None
    fill_value: Optional[float] = 0.0
    impute_numeric_: Optional[SimpleImputer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_columns is None:
            self.numeric_columns = [c for c in X.columns if _is_numeric_series(X[c])]
        else:
            self.numeric_detected = [c for c in self.numeric_columns if c in X.columns]       
        if len(self.numeric_columns) == 0:
            self.imputer_num = None
            return self
        X_num = X[self.numeric_columns]
        if self.strategy == "constant":
            self.impute_numeric_ = SimpleImputer(strategy="constant",                                                
                                                fill_value=self.fill_value,
                                                missing_values=np.nan).fit(X_num)
        else:
            self.impute_numeric_ = SimpleImputer(strategy=self.strategy,
                                                missing_values=np.nan).fit(X_num)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.impute_numeric_ == None:
            return X        
        X = _ensure_dataframe(X)
        X_num = X[self.numeric_columns].astype("float64")
        X_impute = self.impute_numeric_.transform(X_num)
        X.loc[:, self.numeric_columns] = X_impute

        return X

# imputando valores categóricos
@dataclass
class ImputerCategoric(BaseTransform, BaseEstimator):
    strategy: Literal["most_frequent", "constant"] = "most_frequent"
    categoric_columns: Optional[List[str]] = None
    fill_value: str = "other"
    impute_categoric_: Optional[SimpleImputer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_columns is None:
            self.categoric_columns = [c for c in X.columns if _is_categorical_dtype(X[c])]
        else:
            self.categoric_columns = [c for c in self.categoric_columns if c in X.columns]
        if len(self.categoric_columns) == 0:
            return self
        X_cat = X[self.categoric_columns]
        if self.strategy == "constant":
            self.impute_categoric_ = SimpleImputer(strategy="constant",
                                                  missing_values=np.nan,
                                                  fill_value=self.fill_value).fit(X_cat)
        else:
            self.impute_categoric_ = SimpleImputer(strategy=self.strategy,
                                                  missing_values=np.nan).fit(X_cat)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.impute_categoric_ is None:
            return X
        X = _ensure_dataframe(X)
        X_cat = X[self.categoric_columns]
        X_impute = self.impute_categoric_.transform(X_cat)
        X.loc[:, self.categoric_columns] = X_impute

        return X

# valores con menor cantidad en las variables categóricas  
@dataclass
class RareCategoryGroup(BaseTransform, BaseEstimator):
    umbral: float = 0.04
    categoric_columns: Optional[str] = None
    other_label: str = "other"
    keepers_: Dict[str, set] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_columns is None:
            self.categoric_columns = [c for c in X.columns if _is_categorical_dtype(X[c])]
        else:
            self.categoric_columns = [c for c in self.categoric_columns if c in X.columns]
        self.keepers_ = {}
        for c in self.categoric_columns:
            frecuencia = X[c].value_counts(dropna=False)/len(X)
            keep = set(frecuencia[frecuencia > self.umbral].index.to_list())
            self.keepers_[c] = keep
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keepers_ is None:
            return X
        X = _ensure_dataframe(X)
        for c, keep in self.keepers_.items():
            if c in X.columns:
                X[c] = X[c].astype("object").apply(lambda x: x if x in keep else self.other_label)

        return X

# manejo de outliers
@dataclass
class OutliersDetected(BaseTransform):
    numeric_columns: Optional[List[str]] = None
    limites_: Dict[str, tuple[float, float]] = field(default_factory=dict)
    method: Literal["drop", "clip"] = "drop"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_columns is None:
            self.numeric_columns = [c for c in X.columns if _is_numeric_series(X[c])]
        else:
            self.numeric_columns = [c for c in self.numeric_columns if c in X.columns]
        for c in self.numeric_columns:
            q2 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q2
            low = q2 - 1.5*iqr
            high = q3 + 1.5*iqr
            self.limites_[c] = (low, high)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.limites_ is None:
            return X
        if self.method == "drop":
            for c, (low, high) in self.limites_.items():
                if c in X.columns:
                    X = X[(X[c] >= low) & (X[c] <= high)]
        elif self.method == "clip":
             X[self.numeric_columns] = X[self.numeric_columns].astype("float64")
             for c, (low, high) in self.limites_.items():   
                if c in X.columns:
                    X.loc[:, c] = X[c].clip(low, high)

        return X

# escalando los valores numéricos con StandardScaler y MinMaxScaler
@dataclass
class ScalerNum(BaseTransform, BaseEstimator):
    method: Literal["standard", "min_max"] = "standard"
    numeric_columns: Optional[List[str]] = None
    scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = StandardScaler

    def fit(self, X:pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.numeric_columns is None:
            self.numeric_columns = [c for c in X.columns if _is_numeric_series(X[c])]
        else:
            self.numeric_columns = [c for c in self.numeric_columns if c in X.columns]
        X_num = X[self.numeric_columns]
        if self.method == "standard":
            self.scaler_ = StandardScaler().fit(X_num)
        elif self.method == "min_max":
            self.scaler_ = MinMaxScaler().fit(X_num)
        else:
            raise ValueError("Los métodos disponibles son: 'standard', 'min_max'")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            return X
        X = _ensure_dataframe(X)
        X_num = X[self.numeric_columns]
        X_scaler = self.scaler_.transform(X_num)
        X[self.numeric_columns] = X[self.numeric_columns].astype("float64")
        X_scaler_df = pd.DataFrame(X_scaler, index=X.index, columns=self.numeric_columns)
        X.loc[:, self.numeric_columns] = X_scaler_df

        return X

# escalamiento de valores categóricos  
@dataclass
class ScalerCategoric(BaseTransform, BaseEstimator):
    drop_first: Literal["first", "if_binary"] = "first"
    handle_unknown: Literal["ignore", "error"] = "ignore"
    sparse_output: Literal[True, False] = False
    categoric_columns: List[str] = None
    ohe_: Optional[OneHotEncoder] = None

    def fit(self, X:pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)
        if self.categoric_columns is None:
            self.categoric_columns = [c for c in X.columns if _is_categorical_dtype(X[c])]
        else:
            self.categoric_columns = [c for c in self.categoric_columns if c in X.columns]
        X_cat = X[self.categoric_columns]
        self.ohe_ = OneHotEncoder(drop=self.drop_first,
                                  handle_unknown=self.handle_unknown,
                                  sparse_output=self.sparse_output).fit(X_cat)
        return self 

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ohe_ is None:
            return X
        X = _ensure_dataframe(X)
        X_cat = X[self.categoric_columns]
        X_ohe = self.ohe_.transform(X_cat)
        features_names = self.ohe_.get_feature_names_out(self.categoric_columns).tolist()
        X = X.drop(self.categoric_columns, axis=1)
        X.loc[:, features_names] = X_ohe
        
        return X

# definiendo el Pipeline
@dataclass
class Preprocessor(BaseTransform):
    strategy_keep: Literal["first", "last", False] = "first"
    numeric_impute_strategy: Literal["mean","median","most_frequent","constant"] = "mean"
    categorical_impute_strategy: Literal["most_frequent","constant"] = "most_frequent"
    rare_umbral: float = 0.03
    outlier_method: Literal["drop","clip"] = "drop"
    scale_method: Literal["standard", "min_max"] = "standard"
    drop_first: Optional[str] = None

    numeric_columns: Optional[List[str]] = None
    categoric_columns: Optional[List[str]] = None
    pipeline_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = _ensure_dataframe(X)

        detector = ColumnsDetector(numeric_columns=self.numeric_columns,
                                   categoric_columns=self.categoric_columns)
        detector.fit(X)
        self.numeric_columns = detector.numeric_detected
        self.categoric_columns = detector.categoric_detected

        self.pipeline_ = Pipeline(steps=[
            ("drop_duplicates", DetectedDuplicates(keep=self.strategy_keep)),
            ("numeric_imputer", NumericImputer(strategy=self.numeric_impute_strategy,
                                               numeric_columns=self.numeric_columns)),
            ("categorical_imputer", ImputerCategoric(strategy=self.categorical_impute_strategy,
                                                     categoric_columns=self.categoric_columns)),
            ("rare_grouper", RareCategoryGroup(umbral=self.rare_umbral,
                                               categoric_columns=self.categoric_columns)),
            ("outliers", OutliersDetected(numeric_columns=self.numeric_columns,
                                          method=self.outlier_method)),
            ("scaler_num", ScalerNum(method=self.scale_method, numeric_columns=self.numeric_columns)),
            ("scaler_cat", ScalerCategoric(drop_first=self.drop_first,
                                           categoric_columns=self.categoric_columns))
        ])
        self.pipeline_.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _ensure_dataframe(X)
        Xp = self.pipeline_.transform(X)
        if not isinstance(Xp, pd.DataFrame):
            Xp = pd.DataFrame(Xp, index=X.index)
        return Xp
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X = _ensure_dataframe(X)
        self.fit(X, y)
        return self.pipeline_.transform(X)
    
    def fit_transform_xy(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        Xp = self.fit_transform(X, y)
        yp = y.loc[Xp.index]
        return Xp, yp
    
    def fit_transform_concat(self, X: pd.DataFrame, y: pd.Series, y_name: str = None) -> pd.DataFrame:
        Xp, yp = self.fit_transform_xy(X, y)
        if y_name is not None:
            yp = yp.rename(y_name)
        elif yp.name is None:
            yp = yp.rename("target")
        df_xy = pd.concat([Xp, yp], axis=1)
        return df_xy
  
if __name__ == "__main__":
    # df = pd.DataFrame({"nota": [15, 15, 15, 17, 13, 14, 14, 12, np.nan],
    #                    "color": ["rojo", "rojo", "rojo", "azul", "verde", "rojo", "azul", "rojo", "verde"],
    #                    "edad": [12, 12, 12, 14, 15, np.nan, 20, 23, np.nan]})
    # procesor = Preprocessor()
    # procesor.fit(df)
    # procesor.pipeline_.set_params(scaler_num="passthrough",
    #                               outliers="passthrough",
    #                               drop_duplicates="passthrough")
    # procesor.pipeline_.fit(df)
    # Xp = procesor.pipeline_.transform(df)
    # print(Xp)

    df = pd.read_csv("diabetes.csv")
    X = df.drop("Resultado", axis=1)
    y = df["Resultado"]
    procesor = Preprocessor(numeric_impute_strategy="most_frequent")
    # Xp = procesor.fit_transform(X, y)
    # X_y = procesor.fit_transform_concat(X, y)
    # print(X_y)
    
    procesor.fit(X, y)
    procesor.pipeline_.set_params(scaler_num="passthrough")
    Xp= procesor.pipeline_.fit_transform(X, y)
    yp = y.loc[Xp.index]
    X_y_fin = pd.concat([Xp, yp], axis=1)
    print(X_y_fin)
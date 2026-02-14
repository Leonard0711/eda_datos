import os
import sys
from sqlalchemy import create_engine, text

def get_engine():
    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        print("La variable password no está definida")
        sys.exit(1)
    try:
        mysql_engine = create_engine(f"mysql+pymysql://root:{password}@127.0.0.1:3306/practica_sql",
                                     pool_pre_ping=True, pool_recycle=1800, future=True)
        return mysql_engine
    except Exception as e:
        print(f"El error presentado: {e}")
        sys.exit(1)

def solicitar_datos():
    nombre = input("Nombre: ").strip()
    correo = input("Correo: ").strip()
    edad_input = input("Edad: ")
    try:
        edad = int(edad_input)
    except ValueError:
        print("Edad inválida: debe ser un número entero")
        sys.exit(1)
    try:
        if edad < 0:
            raise ValueError("La edad no puede ser negativa")
    except ValueError:
        print("Edad inválida: debe ser un número entero no negativo")
        sys.exit(1)
    if not nombre:
        print("El campo 'nombre' debe estar completo")
        sys.exit(1)
    if not correo:
        print("El campo 'correo' debe estar completo")
        sys.exit(1)
    
    return nombre, correo, edad

def insertar_usuario(engine, nombre, correo, edad):
    t = text("""
            INSERT INTO users_register (nombre, correo, edad)
            VALUES (:n, :c, :e)
            """)
    try:
        with engine.begin() as conn:
            conn.execute(t, {"n": nombre, "c":correo, "e": edad})
        print("Usuario registrado")
    except Exception as e:
        print(f"Error al insertar el usuario: {e}")
        sys.exit(1)

def main():
    engine = get_engine()
    nombre, correo, edad = solicitar_datos()
    insertar_usuario(engine, nombre, correo, edad)

if __name__ == "__main__":
    main()
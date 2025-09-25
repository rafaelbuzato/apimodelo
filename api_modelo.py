import os
import logging
import datetime
import jwt
from functools import wraps
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime,Float
from sqlalchemy.orm import declarative_base, sessionmaker

JWT_SECRET = 'MEUSEGREDOAQUI'
JWT_ALGORITHM = 'HS256'
JWt_ExP_DELTA_SECONDS = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")

DB_URL = 'sqlite:///predictions.db'
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Cria as tabelas no banco de dados
Base.metadata.create_all(engine)

model = joblib.load('modelo_iris.pkl')
logger.info("Modelo carregado com sucesso.")

app = Flask(__name__)
predictions_cache = {}

TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"

def create_token(username):
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWt_ExP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get('password')
    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
       return jsonify({'message': 'Invalid credentials!'}), 401   

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """Endpoint para realizar previsões usando o modelo carregado.
       Corpo {JSON}:
       {
           "sepal_length": float,
           "sepal_width": float,
           "petal_length": float,
           "petal_width": float
       }
    """   
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (KeyError, ValueError) as e:
        logger.error(f"Logs de entrada inválidos: {e}")
        return jsonify({'error': 'Invalid input data', 'details': str(e)}), 400
    
    # Verifica se a previsão já está em cache
    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        predicted_class = predictions_cache[features]
        logger.info(f"Previsão retornada do cache. Features: {features}")
    else:
        # Rodar o modelo
        input_data = np.array([features])  # shape correto (1,4)
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])

        # Armazenar no cache
        predictions_cache[features] = predicted_class
        logger.info(f"Previsão realizada pelo modelo. Features: {features}")

    # Logar no banco
    db = SessionLocal()
    try:
        new_pred = PredictionLog(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            predicted_class=predicted_class,
            created_at=datetime.datetime.utcnow()
        )
        db.add(new_pred)
        db.commit()
        logger.info("Previsão logada no banco de dados.")
    except Exception as e:
        db.rollback()
        logger.error(f"Erro ao salvar previsão no banco: {e}")
        return jsonify({'error': 'Erro ao salvar no banco'}), 500
    finally:
        db.close()

    return jsonify({'predicted_class': predicted_class})

@app.route('/logs', methods=['GET'])
@token_required
def list_predictions():
    """Endpoint para listar todas as previsões armazenadas no banco de dados."""
    limit = int(request.args.get('limit', 10))
    offset = int(request.args.get('offset', 0))
    db = SessionLocal()
    preds = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).offset(offset).limit(limit).all()
    db.close()
    result = []
    for p in preds:
        result.append({
            'id': p.id,
            'sepal_length': p.sepal_length,
            'sepal_width': p.sepal_width,
            'petal_length': p.petal_length,
            'petal_width': p.petal_width,
            'predicted_class': p.predicted_class,
            'created_at': p.created_at.isoformat()
        })
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)
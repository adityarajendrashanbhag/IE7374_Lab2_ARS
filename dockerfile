# Stage 1: Build and Train the Models
FROM python:3.10 AS model_training

WORKDIR /app

COPY src/model_training.py /app/
COPY src/train_rf.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt
RUN python model_training.py
RUN python train_rf.py


# Stage 2: Serve Predictions
FROM python:3.10 AS serving

WORKDIR /app

COPY --from=model_training /app/my_model.keras /app/
COPY --from=model_training /app/tf_scaler.joblib /app/
COPY --from=model_training /app/rf_model.joblib /app/
COPY --from=model_training /app/scaler.joblib /app/
COPY src/main.py /app/
COPY requirements.txt /app/

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY src/templates /app/templates
COPY src/statics /app/statics

EXPOSE 80

CMD ["python", "main.py"]
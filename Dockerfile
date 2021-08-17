FROM python:3.7.5

LABEL "Version" = $VERSION
LABEL "Name" = "mlflow-fastai"

WORKDIR /opt/app
COPY requirements.txt /opt/app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

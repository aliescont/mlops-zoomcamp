FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY starter_ae.py .

ENTRYPOINT ["python", "./starter_ae.py", "--year", "2021", "--month", "4"]



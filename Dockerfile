FROM python:3.11.10-bookworm

ENV PYTHONUNBUFFERED True
ENV APP_HOME /back-end

WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 20 app:app
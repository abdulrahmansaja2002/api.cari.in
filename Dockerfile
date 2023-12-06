FROM python:3.11-alpine

# Path: /app
WORKDIR /app

# Copy all files from the current directory to the container
# except the ones listed in .dockerignore
COPY . .

# Install dependencies
RUN apk --update add --virtual build-dependencies libffi-dev openssl-dev build-base \
  && pip install --upgrade pip \
  && pip install -r requirements.txt \
  && apk del build-dependencies

# set environment variables
ENV PORT 5000
ENV HOST '0.0.0.0'
ENV DEBUG 'False'

# number of results to return
ENV NUM_RESULTS 10
# ENV INDEX_FOLDER 'index_snowball' ==> for snowball stemmer
ENV INDEX_FOLDER 'index' 


# expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "server.py"]
#Bases the docker image off of the libatomsquip/quip image
FROM libatomsquip/quip:latest

#Define who the maintainer is
LABEL maintainer="hensley.derek58@gmail.com".

#Copies current directory into docker image folder name gblearn
COPY . /gblearn

#Sets the directory to newly created gblearn folder
WORKDIR /gblearn

#Install all dependencies
RUN python2 setup.py install
RUN pip install -r requirements.txt




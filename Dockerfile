FROM pypy:slim
ADD . /
CMD [ "pypy3", "./sunfish.py" ]
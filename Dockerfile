FROM python:3.7

#by default, source code is not obfuscated
ARG OBF=false

COPY pyarmor-regcode-1.txt ./pyarmor-regcode-1.txt

RUN pip install pyarmor
RUN pyarmor register pyarmor-regcode-1.txt
 
ADD /src /src
# obfuscate the source code ("src" folder). It creates "dist" folder with obfuscated source code that is copied again to "src" folder.
RUN if [ "$OBF" = "true" ]; then pyarmor obfuscate -r --src="src" sunfish.py;cp -R /dist/. /src; fi;

# remove file license python obfuscator
RUN find pyarmor-regcode-1.txt -delete

CMD python -u /src/sunfish.py "e2e4"
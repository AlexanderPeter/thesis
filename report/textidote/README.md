# How to use textidote

1. Download jar from https://github.com/sylvainhalle/textidote/releases/latest and save it in this directory.

2. Execute one of the following lines for basic verification in this directory:

   ```
   java -jar textidote.jar --no-config --output html ../report.tex > report.html
   java -jar textidote.jar --no-config --output plain ../report.tex
   java -jar textidote.jar --no-config --output singleline ../report.tex
   ```

3. Execute one of the following lines for grammar and spelling check:

   ```
   java -jar textidote.jar
   ```

   It seems the output cannot be saved in a file using a config file. Therefore the following command can be used instead:

   ```
   java -jar textidote.jar --no-config --output html --read-all --dict mydict.txt --encoding cp1252 --ignore sh:nobreak,sh:d:002,sh:nonp --check en ../report.tex > report.html
   ```

4. See https://github.com/sylvainhalle/textidote?tab=readme-ov-file for more.

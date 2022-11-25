wget -c "https://alt.qcri.org/~hmubarak/EGY-MGR-LEV-GLF-2-MSA.zip"
wget -c "https://alt.qcri.org/~hmubarak/EGY-MGR-LEV-GLF-StrongWords.zip"
wget -c "http://alt.qcri.org/~hmubarak/EGY2MSA-sample-correction.zip"
mkdir -p data/DIA2MSA/
mv EGY-MGR-LEV-GLF-2-MSA.zip data/DIA2MSA
mv EGY-MGR-LEV-GLF-StrongWords.zip data/DIA2MSA
mv EGY2MSA-sample-correction.zip data/DIA2MSA
cd data/DIA2MSA
unzip EGY-MGR-LEV-GLF-2-MSA.zip
unzip EGY-MGR-LEV-GLF-StrongWords.zip
unzip EGY2MSA-sample-correction.zip

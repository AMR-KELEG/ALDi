wget -c "https://camel.abudhabi.nyu.edu/madar-parallel-corpus/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip"
mkdir -p data/MADAR
mv MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip data/MADAR/MADAR.zip
cd data/MADAR/
unzip MADAR.zip

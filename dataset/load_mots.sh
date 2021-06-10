TARGET_PATH=$1
CURRENT_PATH=$(pwd)

cd ${TARGET_PATH}

gdown https://drive.google.com/uc?id=1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi -O Task06_Lung.tar
gdown https://drive.google.com/uc?id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL -O Task07_Pancreas.tar
gdown https://drive.google.com/uc?id=1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS -O Task08_HepaticVessel.tar
gdown https://drive.google.com/uc?id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu -O Task03_Liver.tar
gdown https://drive.google.com/uc?id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE -O Task09_Spleen.tar
gdown https://drive.google.com/uc?id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y -O Task10_Colon.tar

NAMES=(Liver HepaticVessel Pancreas Colon Lung Spleen)
TASK_IDS=('03' '08' '07' '10' '06' '09')

for i in {0..5}
  do
	echo ${NAMES[${i}]}

	DIRNAME=$((i+1))${NAMES[${i}]}
	FILENAME=Task${TASK_IDS[${i}]}_${NAMES[${i}]}

	tar -xf ${FILENAME}.tar

	mkdir ${DIRNAME}
	mv ${FILENAME}/* ${DIRNAME}/
	rm -r ${FILENAME}
	rm ${FILENAME}.tar
  done

mv 1Liver 0Liver


#KiTS
mkdir 1Kidney
mkdir 1Kidney/origin
git clone https://github.com/neheller/kits19
cd kits19
#pip3 install -r requirements.txt, maybe you would like to use an env for that
python3 -m starter_code.get_imaging
mv data/* ../1Kidney/origin
cd ..
yes | rm -r kits19

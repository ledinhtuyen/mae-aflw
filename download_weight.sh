FILE_NAME='checkpoint_mae_pretrained.pt'
FILE_ID='1VanxPChGPGcOj93h41lDcjee-7GpZXi9'
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o $FILE_NAME
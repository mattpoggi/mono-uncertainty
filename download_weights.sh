if [ "$#" -eq  "0" ]
then
    echo Usage: $0 M/S/MS
fi

if [ "$1" != "M" ]
then
    echo "M weights only are available for now"
    exit
fi

mkdir -p weights/$1
wget https://www.mediafire.com/file/45oh0877qux6xkd/$1.zip
unzip weights/$1.zip weights/$1/

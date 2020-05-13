if [ "$#" -eq  "0" ]
then
    echo Usage: prepare_monodepth2_engine.sh kitti_datapath
    exit
fi

current=`pwd`

# check if KITTI has already been downloaded
if [ -d $1/2011_09_26 ] && [ -d $1/2011_09_28 ] && [ -d $1/2011_09_29 ] && [ -d $1/2011_09_30 ] && [ -d $1/2011_10_03 ]
then
    echo Found KITTI dataset at $1
else
    echo Download KITTI dataset...
    cd monodepth2
    mkdir $1

    # download archives, unzip and convert to jpg
    wget -i splits/kitti_archives_to_download.txt -P $1
    cd $1
    unzip "*.zip"
    rm "*.zip"
    find $1 -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'

    # return to monodepth2 folder
    cd $current/monodepth2
fi

# check if KITTI accurate ground truth has already been downloaded
if [ -d $1/2011_09_26/*/proj_depth ]
then
    echo Found KITTI accurate ground truth at $1
else
    echo Download KITTI accurate ground truth...

    # download accurate ground truth, unzip and move inside KITTI folders
    wget "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip" -P $1
    cd $1
    unzip data_depth_annotated.zip
    rm data_depth_annotated.zip

    # unzip and move gt to proper folders
    seqs=`cat $current/monodepth2/splits/eigen_benchmark/test_files.txt | cut -d' ' -f1 | cut -d'/' -f2 | uniq`    
    for s in seqs; do
	date=`echo $s | cut -d'_' -f1-3`
        if [ -d train/$s ];
	then
	    mv train/$s/* $1/$date/$s/
	else
	    mv val/$s/* $1/$date/$s/
	fi
    done
    rm -r train
    rm -r val

    # return to monodepth2 folder
    cd $current/monodepth2
fi

# export ground truth
python export_gt_depth.py --data_path $1 --split eigen
python export_gt_depth.py --data_path $1 --split eigen_benchmark

# ready to go!

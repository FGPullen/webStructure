folder="./site.sample/*"
for f in $folder
do 
        echo $f
        wc -l $f
done

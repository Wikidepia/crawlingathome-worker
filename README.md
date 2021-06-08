# Headless Crawling@Home

This script runs https://github.com/TheoCoombes/crawlingathome from linux box. SVG images will most likely be lost in this scenario. Otherwise the script will behave very similar to its Colab source at: https://colab.research.google.com/drive/1P1H-1kc_CFgJE1NOnXywm2fVSoyv2gMW

## TODO
- [ ] Download image every 2000 URLs
- [ ] Save image embedding 
- [ ] Convert images to tfrecords
- [ ] Upload to google drive
- [ ] Shard of chunk (it needs to read all WAT file which will be bad for low ram server)
- [ ] Crawling@Home integration

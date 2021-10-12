from django.db import models

# Create your models here.


class Imagestore(models.Model):
    img_title = models.CharField(max_length=100, default="image Title")
    img_description = models.CharField(max_length=200, default="This is an image description")
    img_uri = models.CharField(max_length=200, default="Image Uri")
    img_usertag = models.CharField(max_length=200, default="Image Tag")
    image = models.ImageField(upload_to='mdeia', blank=False, null=False)
    img_category= models.CharField(max_length=150, default="picture")
    img_xml = models.FileField(blank=True, null=True, upload_to='xmlfiles')
    slug = models.SlugField(null=True, blank=True)

    def __str__(self):
        return self.img_title



import os
import json
from django.core.management.base import BaseCommand
from django.forms.models import modelform_factory
from datetime import datetime
import xml.etree.ElementTree as ET
tree = ET.parse('Bird_2_rucod.xml')
root = tree.getroot()
for child in root:
    print(child.tag, child.attrib)
"""
class Imagestore():

    def __init__(self):
        form = modelform_factory(Imagestore, fields=('img_title', 'img_description', 'img_uri', 'img_usertag','img_category'))
        populated_form = form(data=Bird_1_rucod.json)
        with open('Bird_1_rucod.json') as data_file:
            data = data_file.read()
            jsondata= json.loads(data)
            self.img_title = jsondata['MediaName']
            self.img_description = jsondata['Description']
            self.img_uri = jsondata['MediaUri']
            self.img_usertag = jsondata['Name']
            self.img_category = jsondata['MediaCreationInformation']
            self.save()


<td> <a href="Search/{{obj.object.id}}">{{ obj.object.id }} </a></td>
def __init__(self, json_dict, type_of_gas, normal = True):
        self.img_title = "normal" if normal else "higher"
        self.img_description = img_description
        self.img_uri = json_dict[img_uri][self.img_uri]["tax_co2"]
        self.img_usertag = json_dict[img_usertag][self.img_usertag]["tax"]
        self.img_category = json_dict[img_category][self.img_category]["charge"]
        self.slug = json_dict[slug][self.slug]["updated"]
        reading = Bird_1_rucod.json()
        for object in reading['id']:
            Imagestore.objects.create(img_title=object['MediaName'], img_description=object['Description'],
                                      img_uri=object['MediaUri'], img_usertag=object['Name'], img_category=object['MediaCreationInformation'])
                                      
def import_imagestore_from_file(self):
        data_folder = os.path.join(BASE_DIR, 'import_data', 'jsonfiles')
        for data_file in os.listdir(data_folder):
            with open(os.path.join(data_folder, data_file), encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                for data_object in data:
                    img_title = data_object.get('MediaName', None)
                    img_description = data_object.get('Description', None)
                    img_uri = data_object.get('MediaUri', None)
                    img_usertag= data_object.get('Name', None)
                    image = data_object.get(None)
                    img_category= data_object.get('MediaCreationInformation', None)
                    slug=data_object.get('ImageDescription', None)
                    try:
                        imagestore, created = Imagestore.objects.get_or_create(
                            img_title=img_title,
                            img_description=img_description,
                            img_uri=img_uri,
                            img_usertag=img_usertag,
                            image=image,
                            img_category=img_category,
                            slug=slug,
                        )
                        if created:
                            imagestore.save()
                            display_format = "\nimagestore, {}, has been saved."
                            print(display_format.format(imagestore))
                    except Exception as ex:
                        print(str(ex))
                        msg = "\n\nSomething went wrong saving this movie: {}\n{}".format(img_title, str(ex))
                        print(msg)

"""

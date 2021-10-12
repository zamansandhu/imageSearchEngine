import datetime
from haystack import indexes
from .models import Imagestore


class ImageIndexes(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True, template_name="search/indexes/imgsearcheng/img_text.txt")
    img_category = indexes.CharField(model_attr='img_category')
    content_auto = indexes.EdgeNgramField(model_attr='img_category')

    def get_model(self):
        return Imagestore

    def index_queryset(self, using=None):
        return self.get_model().objects.all()

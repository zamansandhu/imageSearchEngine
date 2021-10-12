from django import forms
from .models import Imagestore


class Imagsearchform(forms.ModelForm):
    class Meta:
        model = Imagestore
        fields = {}


class Imagegraphform(forms.ModelForm):
    class Meta:
        model = Imagestore
        fields = {}


class Imagdetailform(forms.ModelForm):
    class Meta:
        model = Imagestore
        fields = {}


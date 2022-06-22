from pyexpat import model
from statistics import quantiles
from django.db import models
from django.test import modify_settings

# Create your models here.
class Manufacturer(models.Model):
    name = models.CharField(max_length=120)
    location = models.CharField(max_length=120)
    active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.name

class Product(models.Model):
    manufacturer = models.ForeignKey(Manufacturer,
                                                on_delete=models.CASCADE,
                                                related_name='products')
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True,null=True)
    photo = models.ImageField(blank=True,null=True)
    price = models.FloatField()
    shipping_cost = models.FloatField()
    quantity = models.PositiveSmallIntegerField()

    def __str__(self) -> str:
        return self.name
from django.views.generic.detail import DetailView
from django.views.generic.list import ListView

from .models import Product, Manufacturer


class ProductDetailView(DetailView):
    model = Product
    template_name = 'products/templates/product_detail.html'


    
class ProductListView(ListView):
    model = Product
    template_name: str = 'products/templates/product_list.html'
# Create your views here.

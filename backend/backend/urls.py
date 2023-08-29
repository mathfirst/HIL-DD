"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from app01 import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/pdb/", views.getPDBList),
    path("api/confirmpdb/", views.getTimePDB),
    path("api/preference/", views.sendMoleculeList),
    path("api/sendAnnotations/", views.getAnnotations),
    path("api/feedback/", views.evaluation),
    path("test/", views.test),
]


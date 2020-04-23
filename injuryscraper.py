import requests
from bs4 import BeautifulSoup

page = requests.get(
    "https://www.mmamania.com/2014/1/1/5249862/mma-ufc-injuries-list-all-injured-ufc-fighters-2013").text
bs = BeautifulSoup(page, features="html.parser")


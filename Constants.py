DATASET = "Yelp2018"  # Yelp2018  Gowalla  Foursquare  Yelp douban-book
ENCODER = 'hGCN'  # hGCN  Transformer  TransformerLS  gMLP
ABLATION = 'Full'  # Full  w/oImFe  w/oFeTra w/oGlobal w/oAtt w/oConv w/oGraIm
COLD_START = False  # True, False

user_dict = {
    'Yelp': 30887,
    'douban-book': 12859,
    'Gowalla': 18737,
    'Yelp2018': 31668,
    'Foursquare': 7642
}

poi_dict = {
    'Yelp': 18995,
    'douban-book': 22294,
    'Gowalla': 32510,
    'Yelp2018': 38048,
    'Foursquare': 28483
}

POI_NUMBER = poi_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)

print('Dataset:', DATASET, '#User:', USER_NUMBER, '#POI', POI_NUMBER)
print('Encoder: ', ENCODER)
print('ABLATION: ', ABLATION)
print('COLD_START: ', COLD_START)

PAD = 0

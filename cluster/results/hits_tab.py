from tabulate import tabulate

authorative= [[ 0.00578617]
 ,[ 0.01545843]
 ,[ 0.01192455]
 ,[ 0.48491364]
 ,[ 0.12445532]
 ,[ 0.04551569]
 ,[ 0.03994788]
 ,[ 0.0050999 ]
 ,[ 0.01984759]
 ,[ 0.00108189]
 ,[ 0.01617755]
 ,[ 0.01183698]
 ,[ 0.14204622]
 ,[ 0.04221497]
 ,[ 0.01224395]
 ,[ 0.00666919]
 ,[ 0.01478007]
 ,[ 0.        ]]

hub = [[ 0.09025167]
 ,[ 0.03053575]
 ,[ 0.01179884]
 ,[ 0.01282696]
 ,[ 0.08628816]
 ,[ 0.072376  ]
 ,[ 0.13516362]
 ,[ 0.07251028]
 ,[ 0.10399753]
 ,[ 0.03748723]
 ,[ 0.03547686]
 ,[ 0.01938499]
 ,[ 0.10112571]
 ,[ 0.03032891]
 ,[ 0.00609559]
 ,[ 0.14084562]
 ,[ 0.01331413]
 ,[ 0.00019217]]

rules =[ ["Locations$"],["Attractions"],["Flights"],["Hotel_Review"],["Hotelslist1"],["Hotelslist2"],["HotelsNear"]\
                     ,["LastMinute"],["LocalMaps"],["specialOffers"],["Restaurantslists"],["ShowForum"],\
                     ["UserReviews"],["Tourism"],["TravelGuide"],["TravelersChoice"],["VacationRentals"],["UserReview-e"]]
table = []
temp = ["authority"]
print len(authorative)
print len(hub)
print len(rules)

for i in range(len(rules)):
    temp = [rules[i][0],authorative[i][0],hub[i][0]]
    table.append(temp)



header = ["class","authority","hub"]

print header
t = tabulate(table,header,tablefmt="latex",floatfmt=".4f")
print t
from actuarial_tools import *
import prlife



CREATE_PRODUCTS_FROM_DB = True


if CREATE_PRODUCTS_FROM_DB:
	# create products from data base
	prlife.create_products()


# mp = ModelPoint(0, 10, 10, 5, gross_premium=3.33, sum_assured=1000)
# print(ProductA.getCashFlow(DeathBenefit, mp, time_scale=YEAR))

# print(ProductManager.PRODUCTS)
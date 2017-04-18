from actuarial_tools import *


class ProductA(ProductBase, prod_id=1, prod_name="test_product"):
    db1 = DeathBenefit(CashFlowSA(ratio=1.0))
    db2 = DeathBenefit(CashFlowPrem(ratio=1.0))


class ProductB(ProductBase):
    prod_id = 2
    prod_name = "test_product1"
    db1 = DeathBenefit(CashFlowSA(ratio=1.0))
    db2 = DeathBenefit(CashFlowPrem(ratio=1.0))



# mp = ModelPoint(0, 10, 10, 5, gross_premium=3.33, sum_assured=1000)
# print(ProductA.getCashFlow(DeathBenefit, mp, time_scale=YEAR))
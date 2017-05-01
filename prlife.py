"""
    Specialized Module for Pear River Life. Including

    - DataBase connections, at prlife we use PostgreSQL,
        the default assumption schema is `static`, all tables are stored just like prophet
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base, AutomapBase
from sqlalchemy.schema import MetaData
import numpy as np
from actuarial_tools import *


DB_HOST = "10.132.13.245"
DB_DATABASE = "ASS"
DB_ASSUMPTION_SCHEMA = "static"
DB_PORT = 5432
DB_USER = 'developer'
DB_PASSWORD = 'aeris'

ENGINE = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
                       , client_encoding='utf8')

SESSION_MAKER = sessionmaker(bind=ENGINE)
SESSION = SESSION_MAKER()

Base: AutomapBase = automap_base(metadata=MetaData(schema=DB_ASSUMPTION_SCHEMA))
Base.prepare(ENGINE, reflect=True)
Res = automap_base(metadata=MetaData(schema="res"))
Res.prepare(ENGINE, reflect=True)

Channel = Base.classes.channel
Commission = Base.classes.commission
CRTable = Base.classes.crtable
DiscountRate = Base.classes.cng_discountrate
PAD = Base.classes.cng_pad
CommissionOR = Base.classes.commissionor
Expense = Base.classes.expense
ExpenseCIRC = Base.classes.expensecirc
Lapse = Base.classes.lapse
Loading = Base.classes.loading
Universal = Base.classes.parameter_u
ParameterAssumption = Base.classes.parameterassumption
Common = Base.classes.parameterc
CommonStat = Base.classes.prameterstat
ProbTable = Base.classes.probabilitytable
ProphetID = Base.classes.prophetid
SelectionFactor = Base.classes.selectionfactor
SizeAverage = Base.classes.sizeaverage
SpecialTable = Base.classes.sptable
PlanNameMap = Res.classes.plan_name_map


def get_prophet_id(idx: int, session=SESSION) -> str:
    """ get prophet id from 精算代码 """
    return session.query(ProphetID).filter_by(plan_id=idx).first().prophet_id

def get_plan_name(idx: int, session=SESSION) -> str:
    return session.query(PlanNameMap).filter_by(plan_id=idx).first().plan_name

def get_prob_table(table_index, session=SESSION) -> np.ndarray:
    """ get table array from table index """
    return np.array(session.query(ProbTable).filter_by(tbl_id=table_index).first().tbl)


def get_ul_parameter(prophet_id: str, session=SESSION) -> Universal:
    parameter: Universal = session.query(Universal).filter_by(prophet_id=prophet_id).first()
    return parameter


def get_non_ul_prarameter(prophet_id: str, session=SESSION):
    return session.query(Common).join(CommonStat, Common.prophet_id==CommonStat.prophet_id).filter_by(prophet_id=prophet_id).first()


def gen_sb_get_cashflow(prophet_id: str, session=SESSION):
    row = session.query(SpecialTable).filter_by(prophet_id=prophet_id, var='SB').first()
    if row.type == 1:
        dtype = CashFlowGetterFactory.PAYMENT_TREM
    elif row.type == 2:
        dtype = CashFlowGetterFactory.AGE
    elif row.type == 3:
        dtype = CashFlowGetterFactory.PAYMENT_TREM_AGE
    else:
        raise ValueError(f"type={row.type}")
    d = row.tbl
    def normalize(dd: dict):
        for k, v in dd.items():
            if not isinstance(v, dict):
                dd[k] = [x/1000 for x in v]
            else:
                dd[k] = normalize(v)
        return dd
    return CashFlowGetterFactory(dict_type=dtype, ratio_dict=normalize(d))
    

def create_products(session=SESSION):
    # nonlocal ProductManager
    # nonlocal ProductBase
    # nonlocal DeathBenefit
    # nonlocal CriticalIllnessBenefit
    # nonlocal AccidentBenefit
    # nonlocal CashFlowSA
    # nonlocal CashFlowPrem

    for r in session.query(ProphetID).filter(ProphetID.plan_id<30000000).all():
        plan_id, prophet_id = r.plan_id, r.prophet_id
        plan_name = get_plan_name(plan_id)
        parameter = get_non_ul_prarameter(prophet_id=prophet_id)
        name_space = OrderedDict(prod_id=plan_id, prod_name=plan_name)
        if parameter.db_pm_pc > 0:
            db_prem = DeathBenefit(CashFlowPrem(ratio=float(parameter.db_pm_pc / 100)))
            name_space['db_prem'] = db_prem
        if parameter.db_sa_pc > 0:
            db_sa = DeathBenefit(CashFlowSA(ratio=float(parameter.db_sa_pc / 100)))
            name_space['db_sa'] = db_sa
        if parameter.crb_pm_pc > 0:
            crb_pm = CriticalIllnessBenefit(CashFlowPrem(ratio=float(parameter.crb_pm_pc / 100)))
            name_space['crb_pm'] = crb_pm
        if parameter.crb_sa_pc > 0:
            crb_sa = CriticalIllnessBenefit(CashFlowSA(ratio=float(parameter.crb_sa_pc / 100)))
            name_space['crb_sa'] = crb_sa
        if parameter.adb_gen_unit_pc > 0:
            accident1 = AccidentBenefit(CashFlowSA(ratio=float(parameter.adb_gen_unit_pc / 100)))
            name_space['accident1'] = accident1
        if parameter.adb_pt_unit_pc > 0:
            accident2 = AccidentBenefit(CashFlowSA(ratio=float(parameter.adb_pt_unit_pc / 100)))
            name_space['accident2'] = accident2
        if parameter.adb_van_unit_pc > 0:
            accident3 = AccidentBenefit(CashFlowSA(ratio=float(parameter.adb_van_unit_pc / 100)))
            name_space['accident3'] = accident3
        if parameter.adb_avia_unit_pc > 0:
            accident4 = AccidentBenefit(CashFlowSA(ratio=float(parameter.adb_avia_unit_pc / 100)))
            name_space['accident4'] = accident4
        if parameter.mb_pm_pc > 0:
            mb_pm = MaturityBenefit(CashFlowPrem(ratio=float(parameter.mb_pm_pc / 100)))
            name_space['mb_pm'] = mb_pm
        if parameter.mb_sa_pc > 0:
            mb_sa = MaturityBenefit(CashFlowSA(ratio=float(parameter.mb_sa_pc / 100)))
            name_space['mb_sa'] = mb_sa
        if parameter.sb_type > 0:
            cf_type = CashFlowSA() if parameter.sb_type == 1 else CashFlowPrem()
            sb = SurvivalBenefit(cash_flow_type=cf_type, getCashFlow=gen_sb_get_cashflow(prophet_id=prophet_id))
            name_space['sb'] = sb
        
        type(prophet_id, (ProductBase,), name_space)
        
            

if __name__ == '__main__':
    prophet_id=get_prophet_id(10313001)
    # print(get_non_ul_prarameter(prophet_id=prophet_id).__dict__)
    # print(get_plan_name(10313001))
    # from actuarial_tools import *
    # create_products()
    # print(ProductManager.PRODUCTS)
    create_products()
    p = ProductManager.PRODUCTS[10412001]
    mp = ModelPoint(sex=0, age=10, policy_term=10, payment_term=5, policy_year=2, sum_assured=5000, gross_premium=1000)
    print(p.getCashFlow(SurvivalBenefit, mp, from_init=True, time_scale=YEAR))
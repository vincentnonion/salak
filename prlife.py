from typing import List, Optional
from collections import OrderedDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.automap import automap_base, AutomapBase
from sqlalchemy.schema import MetaData
import numpy as np
import pdb
from actuarial_tools import *


"""
    Specialized Module for Pear River Life. Including

    - DataBase connections, at prlife we use PostgreSQL,
        the default assumption schema is `static`, all tables are stored just like prophet
"""


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

# class PRProductInfo(ProductInfo):

#     _DESIGN_TYPE = {
#         "普通型": DesignType.PLAIN,
#         "万能型": DesignType.UNIVERSAL,
#         "分红型": DesignType.PARTICIPATE,
#         "投资连结型": DesignType.INVEST
#     }

#     _PRODUCT_TYPE = {
#         "定期寿险": ProductType.TERM_LIFE,
#         "两全寿险": ProductType.ENDOWMENT,
#         "年金保险": ProductType.ANNUITY,
#         "意外伤害保险": ProductType.ACCIDENT,
#         "终身寿险": ProductType.WHOLE_LIFE,
#         "疾病保险": ProductType.HEALTH,
#         "医疗保险": ProductType.HEALTH,
#     }

#     def __init__(self, idx, name=None, designType=None, productType=None, addOn=None, forGroup=False,
#                  session=SESSION):
#         with ENGINE.connect() as conn:
#             rst = conn.execute(f"SELECT plan_name, design_type, liab_type_b FROM public.plan_info WHERE plan_id={idx}").fetchone()

#         prophet_id = _get_prophet_id(idx=idx, session=session)
#         if name is None:
#             name = rst[0]
#         if designType is None:
#             designType = self._DESIGN_TYPE[rst[1]]
#         if productType is None:
#             productType = self._PRODUCT_TYPE[rst[2]]
#         if addOn is None:
#             addOn = self._is_addon(prophet_id, designType=designType)

#         super().__init__(idx=idx, name=name, designType=designType, productType=productType,
#                          addOn=addOn, forGroup=forGroup)
#         self.prophet_id: str = prophet_id

#     @staticmethod
#     def _is_addon(prophet_id: str, designType: DesignType, session=SESSION) -> bool:
#         """ 检查是否为附加险 """
#         if designType == DesignType.UNIVERSAL:
#             return int(session.query(Universal).filter_by(prophet_id=prophet_id).first().rider_flag) == 1
#         else:
#             return int(session.query(Common).filter_by(prophet_id=prophet_id).first().rider_flag) == 1


# class PRBenefit(Benefit):
#     """
#         ADD CV BASED BENEFIT SUPPORT
#     """

#     def __init__(self, ratio: float, *, benefitType: BenefitType, benefitBaseType: BenefitBaseType,
#                  timeType: Optional[TimeType] = None, prophet_id=None, session=SESSION):
#         super().__init__(ratio=ratio, benefitType=benefitType, benefitBaseType=benefitBaseType, timeType=timeType)
#         if benefitBaseType != BenefitBaseType.MAX_CASH_VALUE_PREMIUM:
#             self._cv_raw = None
#             self._cv_type = None
#         else:
#             assert prophet_id is not None
#             row = session.query(SpecialTable).filter_by(prophet_id=prophet_id, var="CV").first()
#             self._cv_raw = row.tbl
#             self._cv_type = row.type  # 1 for PPP, 2 for AGE, 3 for PPP_AGE
#             self._cv_key_max = max((int(x) for x in self._cv_raw.keys()))

#     def _get_cv(self, mp: ModelPoint) -> np.ndarray:
#         if self._cv_type == 1:
#             # 1 for PPP
#             key = str(mp.paymentTerm) if mp.paymentTerm <= self._cv_key_max else str(self._cv_key_max)
#             arr: List[float] = self._cv_raw[key]
#         elif self._cv_type == 2:
#             # 2 for AGE
#             key = str(mp.age) if mp.age <= self._cv_key_max else str(self._cv_key_max)
#             arr: List[float] = self._cv_raw[key]
#         elif self._cv_type == 3:
#             # 3 for PPP_AGE
#             key = str(mp.paymentTerm) if mp.paymentTerm <= self._cv_key_max else str(self._cv_key_max)
#             arr: List[float] = self._cv_raw[key][str(mp.age)]
#         else:
#             raise ValueError(f"static.sptable.type={self._cv_type} Not Supported")

#         return np.array(arr[: mp.benefitTerm]) / 1000.0

#     def getBenefit(self, mp: ModelPoint, *, timeScale: TimeScale=TimeScale.MONTH, fromInit: bool=False) -> np.ndarray:
#         if self.benefitBaseType == BenefitBaseType.MAX_CASH_VALUE_PREMIUM:
#             cv = self._get_cv(mp)
#             premium = mp.premium if mp.premium is not None else 1.0
#             base = np.concatenate((np.arange(1, mp.paymentTerm + 1),
#                                    np.full(mp.benefitTerm - mp.paymentTerm, mp.paymentTerm * premium,
#                                            float))) * self.ratio
#             base = np.fmax(cv * premium * self.ratio, base)
#             if timeScale == TimeScale.YEAR:
#                 if fromInit:
#                     return base
#                 else:
#                     return base[mp.polYrIndex:]
#             elif timeScale == TimeScale.MONTH:

#                 if self.benefitType != BenefitType.SURVIVE:
#                     arr = base.repeat(12)
#                 else:
#                     # 年度末给付生存责任
#                     arr = np.zeros(base.size * 12)
#                     arr[np.arange(base.size) * 12 + 11] = base

#                 if fromInit:
#                     return arr
#                 else:
#                     return arr[mp.polMthIndex]
#         else:
#             return super().getBenefit(mp=mp, timeScale=timeScale, fromInit=fromInit)

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
        
        type(prophet_id, (ProductBase,), name_space)
        
            

if __name__ == '__main__':
    prophet_id=get_prophet_id(10313001)
    print(get_non_ul_prarameter(prophet_id=prophet_id).__dict__)
    print(get_plan_name(10313001))
    from actuarial_tools import *
    create_products()
    print(ProductManager.PRODUCTS)
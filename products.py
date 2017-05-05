from actuarial_tools import *
import prlife

CREATE_PRODUCTS_FROM_DB = False

if CREATE_PRODUCTS_FROM_DB:
    # create products from data base
    prlife.create_products()


class 乐居保(ProductBase, prod_id=10112001):
    prod_name = "珠江乐居保减额定期寿险"
    db = DeathBenefit(CashFlowSA(),
                      getCashFlow=CashFlowGetterFactory.arithmeticDescendingCashFlowGetter(ratio=1.0))


class 定寿(ProductBase, prod_id=10113001):
    prod_name = "珠江定期寿险"
    db = DeathBenefit(CashFlowSA(ratio=1.0))


class 附加豁免定寿(ProductBase, prod_id=10113001):
    prod_name = "珠江附加豁免保费定期寿险"
    db = DeathBenefit(CashFlowSA(),
                      getCashFlow=CashFlowGetterFactory.waiverCashFlowGetter(discount_rate=0.025))


class 附加定寿(ProductBase, prod_id=10123001):
    prod_name = "珠江附加定期寿险"
    db = DeathBenefit(CashFlowSA(ratio=1.0))


class 保驾护航(ProductBase, prod_id=10313001):
    prod_name = "珠江保驾护航两全保险"

    db = DeathBenefit(CashFlowPrem(ratio=1.1))
    adb = AccidentBenefit(CashFlowSA(ratio=1.0))
    adb_public_transport = AccidentBenefit(CashFlowSA(ratio=2.0))
    adb_private_vehicle = AccidentBenefit(CashFlowSA(ratio=5.0))
    adb_air = AccidentBenefit(CashFlowSA(ratio=9.0))
    mb = MaturityBenefit(CashFlowPrem(ratio=1.1))


class 保驾护航2017(ProductBase, prod_id=10313002):
    prod_name = "珠江保驾护航两全保险（2017版）"

    @staticmethod
    def _db_cf_getter(mp: ModelPoint, from_init: bool, time_scale: TimeScale) -> ndarray:
        """
        珠江保驾护航两全保险（2017版）死亡责任与年龄有关

        ==============================  ============
        身故或全残时被保险人的到达年龄       给付比例系数
        ==============================  ============
        18周岁至40周岁                     160%
        41周岁至60周岁                     140%
        61周岁及以上                       120%
        ==============================  ============
        """
        ages = arange(mp.age, mp.age + mp.policy_term)
        rates = np.piecewise(ages, [np.logical_and(ages >= 18, ages <= 40),
                                    np.logical_and(ages >= 41, ages <= 60),
                                    ages > 60], [1.6, 1.4, 1.2])
        if time_scale is TimeScale.MONTH:
            rates = repeat(rates, 12) if from_init else repeat(rates, 12)[mp.index_policy_month:]
        elif time_scale is TimeScale.YEAR and not from_init:
            rates = rates[mp.index_policy_year:]
        return rates

    db = DeathBenefit(CashFlowPrem(), getCashFlow=_db_cf_getter)
    adb = AccidentBenefit(CashFlowSA(ratio=1.0))
    adb_public_transport = AccidentBenefit(CashFlowSA(ratio=2.0))
    adb_private_vehicle = AccidentBenefit(CashFlowSA(ratio=5.0))
    adb_air = AccidentBenefit(CashFlowSA(ratio=9.0))
    mb = MaturityBenefit(CashFlowPrem(ratio=1.1))


class 富多多(ProductBase, prod_id=10411001):
    prod_name = "珠江富多多年金保险"

    db = DeathBenefit(CashFlowMaxPremCV(ratio=1.0))
    sb = SurvivalBenefit(CashFlowSA(),
                         getCashFlow=CashFlowGetterFactory(dict_type=CashFlowGetterFactory.PAYMENT_TREM,
                                                           ratio_dict={1: array([0] + [0.04] * 8 + [0])}))
    mb = MaturityBenefit(CashFlowSA(ratio=1.0))


class 富多多B(ProductBase, prod_id=10411004):
    prod_name = "珠江富多多年金保险（B款）"

    db = DeathBenefit(CashFlowMaxPremCV(ratio=1.0))
    sb = SurvivalBenefit(CashFlowSA(),
                         getCashFlow=CashFlowGetterFactory(dict_type=CashFlowGetterFactory.PAYMENT_TREM,
                                                           ratio_dict={1: array([0] * 5 + [0.04] * 5)}))
    mb = MaturityBenefit(CashFlowSA(ratio=1.0))


class 乐多多(ProductBase, prod_id=10412001):
    prod_name = "珠江乐多多年金保险"

    db = DeathBenefit(CashFlowMaxPremCV(ratio=1.0))
    sb = SurvivalBenefit(CashFlowSA(),
                         getCashFlow=CashFlowGetterFactory(dict_type=CashFlowGetterFactory.PAYMENT_TREM,
                                                           ratio_dict={3: array([0] * 5 + [0.04] * 5),
                                                                       5: array([0] * 5 + [0.04] * 5)
                                                                       }))
    mb = MaturityBenefit(CashFlowSA(ratio=1.0))


class 鸿运年年(ProductBase, prod_id=10413001):
    prod_name = "珠江鸿运年年年金保险"

    db = DeathBenefit(CashFlowPrem(ratio=1.0))

    @staticmethod
    def _sb_cf_getter(mp: ModelPoint, *, from_init: bool=False, time_scale: TimeScale=MONTH):
        """
        #. 生存保险金
            自本合同生效后的首个保单周年日起至被保险人年满59 周岁之后的首个保单周年日，若被保险
            人在每个保单周年日时仍生存且本合同仍然有效，本公司将于每个保单周年日按基本保险金额的10%
            给付生存保险金。
        #. 祝寿保险金
            至被保险人年满60周岁之后的首个保单周年日，若被保险人生存且本合同仍然有效，本公司将
            按基本保险金额的60%给付1次祝寿保险金。
        #. 养老保险金
            自被保险人年满61 周岁之后的首个保单周年日起，若被保险人在每个保单周年日时仍生存且本
            合同仍然有效，本公司将于每个保单周年日按基本保险金额的20%给付养老保险金，直至被保险人
            年满79 周岁之后的首个保单周年日。
         """

        age_arr = arange(mp.age, mp.age + mp.policy_term)
        year_ratio = np.piecewise(age_arr, [age_arr < 60, age_arr == 60, np.logical_and(age_arr > 60, age_arr < 80)],
                                  [0.1, 0.6, 0.2])
        if time_scale is YEAR:
            return year_ratio if from_init else year_ratio[mp.index_policy_year:]
        elif time_scale is MONTH:
            return repeat(year_ratio, 12)[mp.index_policy_month:] if not from_init else repeat(year_ratio, 12)

    sb = SurvivalBenefit(CashFlowSA(), getCashFlow=_sb_cf_getter)
    mb = MaturityBenefit(CashFlowPrem(ratio=1.0))


class 福多多(ProductBase, prod_id=10513002):
    prod_name = "珠江福多多防癌疾病保险"

    db = DeathBenefit(CashFlowPrem(ratio=1.0))
    cib = CriticalIllnessBenefit(CashFlowSA(ratio=1.0))


class 康爱一生(ProductBase, prod_id=10513003):
    prod_name = "珠江康爱一生恶性肿瘤疾病保险"

    db = DeathBenefit(CashFlowPrem(ratio=1.0))
    cib = CriticalIllnessBenefit(CashFlowSA(ratio=1.0))


# fixme
class 康佑(ProductBase, prod_id=10513004):
    prod_name = "珠江康佑终身重大疾病保险"

    db = DeathBenefit(CashFlowSA(ratio=1.0))
    cib = CriticalIllnessBenefit(CashFlowSA(ratio=1.0))
    e_cib = OtherIllnessBenefit(CashFlowSA(ratio=0.3))


class 附加豁免重疾(ProductBase, prod_id=10522001):
    prod_name = "珠江附加豁免保险费重大疾病保险"

    cib = CriticalIllnessBenefit(CashFlowSA(),
                                 getCashFlow=CashFlowGetterFactory.waiverCashFlowGetter(discount_rate=0.025))


class 附加重疾(ProductBase, prod_id=10523001):
    prod_name = "珠江附加重大疾病保险"
    cib = CriticalIllnessBenefit(CashFlowSA(ratio=1.0))


class 康瑞人生(ProductBase, prod_id=20213001):
    prod_name = "珠江康瑞人生终身寿险（分红型）"
    db = DeathBenefit(CashFlowSA(ratio=1.0))


class 利多多(ProductBase, prod_id=20311001):
    prod_name = "珠江利多多两全保险（分红型）"
    db = DeathBenefit(CashFlowSA(ratio=1.0))
    mb = MaturityBenefit(CashFlowSA(ratio=1.0))
    adb = AccidentBenefit(CashFlowSA(ratio=1.0))
    adb_transport = AccidentBenefit(CashFlowSA(ratio=1.0))

    @staticmethod
    def _sb_cf_getter(mp: ModelPoint, from_init: bool, time_scale: TimeScale) -> ndarray:
        """
        生存保险金

        至本合同生效后的第3个保单周年日，若被保险人生存且本合同仍然有效，
        本公司将按已交保险费的5%给付1次生存保险金。
        """
        year_ratio = zeros(mp.policy_term)
        year_ratio[2] = 0.05
        if time_scale is YEAR:
            return year_ratio if from_init else year_ratio[mp.index_policy_year:]
        elif time_scale is MONTH:
            return repeat(year_ratio, 12) if from_init else repeat(year_ratio, 12)[mp.index_policy_month:]

    sb = SurvivalBenefit(CashFlowPrem(), getCashFlow=_sb_cf_getter)


class 喜多多(ProductBase, prod_id=20312001):
    """
    #. 满期保险金
        被保险人生存至保险期间届满，本公司将按本合同累计已交保险费给付满期保险金，本合同随之终止。
    #. 生存保险金
        自本合同生效之日起，被保险人生存至每满三个保单年度的保单周年日，
        本公司将按基本保险金额的一定比例（详见下表）给付1次生存保险金，直至保险期间届满。

            ==============   ============  =============
            交费期间和方式	  3年限期年交	 5年限期年交
            ==============   ============  =============
            给付比例	        15%	         21%
            ==============   ============  =============
    #. 疾病身故保险金
        被保险人在本合同保险责任有效期内因疾病身故，本公司将按本合同累计已交保险费的105%给付身故保险金，本合同终止。
    #. 意外身故保险金
        被保险人在保险期间内遭受意外伤害，并自该意外伤害发生之日起180日内因该意外伤害为直接原因导致身故，
        本公司将按本合同累计已交保险费的200%给付意外身故保险金，本合同随之终止。
    #. 特定交通工具意外身故保险金
        被保险人驾驶或乘坐特定交通工具（指被保险人以乘客身份乘坐火车、轮船、公共汽车或民航班机，
        驾驶或乘坐私家车、单位公务或商务用车）期间，在交通工具内发生意外伤害，并自该意外伤害发生之日起180日内，
        因该意外伤害为直接原因导致身故，本公司除给付意外身故保险金外，
        还将按本合同累计已交保险费给付特定交通工具意外身故保险金，本合同随之终止。
    """
    prod_name = "珠江喜多多两全保险（分红型）"

    db = DeathBenefit(CashFlowPrem(ratio=1.05))
    adb = AccidentBenefit(CashFlowPrem(ratio=0.95))
    adb_transport = AccidentBenefit(CashFlowPrem(ratio=1.0))

    @staticmethod
    def _sb_cf_getter(mp: ModelPoint, from_init: bool, time_scale: TimeScale) -> ndarray:
        assert mp.payment_term in (3, 5)
        ratio = 0.15 if mp.payment_term == 3 else 0.21
        year_ratio = zeros(mp.policy_term)
        year_ratio[arange(2, mp.policy_term, 3)] = ratio
        if time_scale is YEAR:
            return year_ratio if from_init else year_ratio[mp.index_policy_year:]
        elif time_scale is MONTH:
            return repeat(year_ratio, 12) if from_init else repeat(year_ratio, 12)[mp.index_policy_month:]

    sb = SurvivalBenefit(CashFlowSA(), getCashFlow=_sb_cf_getter)


class 尊享人生(ProductBase, prod_id=20313001):
    prod_name = "珠江尊享人生两全保险（分红型）"

    # TODO


class 康逸人生(ProductBase, prod_id=20313002):
    prod_name = "珠江康逸人生两全保险（分红型）"

    # TODO


# TODO 万能302xxxx
# TODO 万能303xxxx

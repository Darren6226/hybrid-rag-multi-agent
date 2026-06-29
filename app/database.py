import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URI

logger = logging.getLogger(__name__)

Base = declarative_base()

# 懒加载：engine / Session 在首次使用时才创建
_engine = None
_Session = None


def _get_session():
    """获取数据库 Session（懒加载 engine 和 sessionmaker）"""
    global _engine, _Session
    if _Session is None:
        _engine = create_engine(DATABASE_URI)
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine)
    return _Session()

class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)

class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)
    customer_name = Column(String(50))
    contact_info = Column(String(50))
    region = Column(String(50))
    customer_type = Column(String(50))

class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(50))
    category = Column(String(50))
    unit_price = Column(Float)
    stock_level = Column(Integer)

class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)
    competitor_name = Column(String(50))
    region = Column(String(50))
    market_share = Column(Float)

def init_seed_data():
    session = _get_session()
    try:
        existing_count = session.query(SalesData).count()
        if existing_count > 0:
            logger.info("数据库已有 %d 条销售记录，跳过初始化。", existing_count)
            return

        logger.info("正在初始化数据库种子数据...")
        customers = [
            CustomerInformation(customer_id=1, customer_name="小米科技有限责任公司", contact_info="10010", region="北京", customer_type="企业客户"),
            CustomerInformation(customer_id=2, customer_name="华为技术有限公司", contact_info="10086", region="深圳", customer_type="企业客户"),
            CustomerInformation(customer_id=3, customer_name="苹果公司", contact_info="400-666-8800", region="美国", customer_type="跨国企业"),
        ]
        session.add_all(customers)
        
        products = [
            ProductInformation(product_id=1, product_name="智能手机", category="电子产品", unit_price=2999.0, stock_level=500),
            ProductInformation(product_id=2, product_name="笔记本电脑", category="电子产品", unit_price=5999.0, stock_level=200),
            ProductInformation(product_id=3, product_name="智能手表", category="可穿戴设备", unit_price=1299.0, stock_level=300),
        ]
        session.add_all(products)
        
        competitors = [
            CompetitorAnalysis(competitor_id=1, competitor_name="三星电子", region="韩国", market_share=22.5),
            CompetitorAnalysis(competitor_id=2, competitor_name="OPPO", region="中国", market_share=18.3),
            CompetitorAnalysis(competitor_id=3, competitor_name="vivo", region="中国", market_share=15.7),
        ]
        session.add_all(competitors)
        
        sales = [
            SalesData(sales_id=1, product_id=1, employee_id=101, customer_id=1, sale_date="2024-01-15", quantity=10, amount=29990.0, discount=0.05),
            SalesData(sales_id=2, product_id=2, employee_id=102, customer_id=2, sale_date="2024-01-20", quantity=5, amount=29995.0, discount=0.10),
            SalesData(sales_id=3, product_id=3, employee_id=103, customer_id=3, sale_date="2024-01-25", quantity=15, amount=19485.0, discount=0.00),
        ]
        session.add_all(sales)
        session.commit()
        logger.info("种子数据初始化完成")
    except Exception as e:
        session.rollback()
        logger.error("种子数据初始化失败: %s", e)
    finally:
        session.close()

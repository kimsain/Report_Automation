import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 데이터베이스 연결 정보
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# 데이터베이스 설정
Base = declarative_base()
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# 데이터베이스 모델 정의
class Region(Base):
    __tablename__ = 'regions'
    
    id = Column(Integer, primary_key=True)
    region_code = Column(Integer, index=True)  # 영역 구분
    
    # 관계 정의
    festival_stats = relationship("FestivalStats", back_populates="region")
    population_stats = relationship("PopulationStats", back_populates="region")
    sales_stats = relationship("SalesStats", back_populates="region")
    hourly_stats = relationship("HourlyStats", back_populates="region")
    inflow_stats = relationship("InflowStats", back_populates="region")
    
    def __repr__(self):
        return f"<Region(region_code={self.region_code})>"

class FestivalStats(Base):
    __tablename__ = 'festival_stats'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'))
    period_type = Column(String(20))  # 축제 전/축제 기간
    start_date = Column(String(20))
    end_date = Column(String(20))
    sales_amount = Column(Float)
    sales_increase_rate = Column(Float)
    main_business_type = Column(String(50))
    visitors = Column(Float)
    visitor_increase_rate = Column(Float)
    main_age_group = Column(String(20))
    main_time_period = Column(String(20))
    
    # 관계 정의
    region = relationship("Region", back_populates="festival_stats")
    
    def __repr__(self):
        return f"<FestivalStats(period_type={self.period_type}, visitors={self.visitors})>"

class PopulationStats(Base):
    __tablename__ = 'population_stats'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'))
    gender = Column(String(10))
    age_group = Column(String(10))
    period_type = Column(String(20))
    visitors = Column(Float)
    
    # 관계 정의
    region = relationship("Region", back_populates="population_stats")
    
    def __repr__(self):
        return f"<PopulationStats(gender={self.gender}, age_group={self.age_group}, visitors={self.visitors})>"

class SalesStats(Base):
    __tablename__ = 'sales_stats'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'))
    business_type = Column(String(50))
    sales_amount = Column(Float)
    
    # 관계 정의
    region = relationship("Region", back_populates="sales_stats")
    
    def __repr__(self):
        return f"<SalesStats(business_type={self.business_type}, sales_amount={self.sales_amount})>"

class HourlyStats(Base):
    __tablename__ = 'hourly_stats'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'))
    period_type = Column(String(20))
    time_period = Column(String(10))
    sales_amount = Column(Float)
    visitors = Column(Float)
    
    # 관계 정의
    region = relationship("Region", back_populates="hourly_stats")
    
    def __repr__(self):
        return f"<HourlyStats(time_period={self.time_period}, visitors={self.visitors})>"

class InflowStats(Base):
    __tablename__ = 'inflow_stats'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'))
    inflow_region_code = Column(String(20))
    visitors = Column(Float)
    inflow_type = Column(String(10))  # 관내/관외
    region_code = Column(String(20))
    region_name = Column(String(50))
    province_name = Column(String(50))
    
    # 관계 정의
    region = relationship("Region", back_populates="inflow_stats")
    
    def __repr__(self):
        return f"<InflowStats(region_name={self.region_name}, visitors={self.visitors})>"

class MapData(Base):
    __tablename__ = 'map_data'
    
    id = Column(Integer, primary_key=True)
    center_lat = Column(Float)
    center_lon = Column(Float)
    total = Column(Float)
    region_code = Column(Integer)
    
    def __repr__(self):
        return f"<MapData(center_lat={self.center_lat}, center_lon={self.center_lon})>"

class HistogramData(Base):
    __tablename__ = 'histogram_data'
    
    id = Column(Integer, primary_key=True)
    geometry = Column(Text)
    total = Column(Float)
    region_code = Column(Integer)
    
    def __repr__(self):
        return f"<HistogramData(total={self.total})>"

class TmapData(Base):
    __tablename__ = 'tmap_data'
    
    id = Column(Integer, primary_key=True)
    destination_name = Column(String(100))
    destination_x = Column(Float)
    destination_y = Column(Float)
    visit_count = Column(Integer)
    stay_minutes = Column(Float)
    second_destination_addr = Column(String(100))
    second_destination_x = Column(Float)
    second_destination_y = Column(Float)
    region_code = Column(Integer)
    destination_type = Column(String(50))
    
    def __repr__(self):
        return f"<TmapData(destination_name={self.destination_name}, visit_count={self.visit_count})>"

# 데이터 로드 및 처리 함수
def load_and_process_data():
    # 데이터베이스 테이블 생성
    Base.metadata.create_all(engine)
    session = Session()
    
    # 데이터 파일 경로
    data_dir = './data'
    
    # 축제분석_현황판.csv 로드
    festival_df = pd.read_csv(os.path.join(data_dir, '축제분석_현황판.csv'))
    
    # 성연령별_방문인구.csv 로드
    population_df = pd.read_csv(os.path.join(data_dir, '성연령별_방문인구.csv'))
    
    # 업종별 매출.csv 로드
    sales_df = pd.read_csv(os.path.join(data_dir, '업종별 매출.csv'))
    
    # 시간대별 인구 및 매출.csv 로드
    hourly_df = pd.read_csv(os.path.join(data_dir, '시간대별 인구 및 매출.csv'))
    
    # 유입인구.csv 로드
    inflow_df = pd.read_csv(os.path.join(data_dir, '유입인구.csv'))
    
    # map_input_for_powerbi.csv 로드
    map_df = pd.read_csv(os.path.join(data_dir, 'map_input_for_powerbi.csv'))
    
    # histo_for_powerbi.csv 로드
    histo_df = pd.read_csv(os.path.join(data_dir, 'histo_for_powerbi.csv'))
    
    # tmap_fin.csv 로드
    tmap_df = pd.read_csv(os.path.join(data_dir, 'tmap_fin.csv'))
    
    # 고유한 영역 구분 코드 추출
    region_codes = set()
    for df in [festival_df, population_df, sales_df, hourly_df, inflow_df, map_df, histo_df, tmap_df]:
        if '영역 구분' in df.columns:
            region_codes.update(df['영역 구분'].unique())
    
    # Region 테이블에 데이터 삽입
    region_id_map = {}  # 영역 구분 코드와 ID 매핑을 위한 딕셔너리
    for code in region_codes:
        region = Region(region_code=int(code))
        session.add(region)
    session.commit()
    
    # 영역 구분 코드와 ID 매핑 생성
    for region in session.query(Region).all():
        region_id_map[region.region_code] = region.id
    
    # 축제분석_현황판.csv 데이터 삽입
    for _, row in festival_df.iterrows():
        festival_stat = FestivalStats(
            region_id=region_id_map[int(row['영역 구분'])],
            period_type=row['구분'],
            start_date=row['시작'],
            end_date=row['종료'],
            sales_amount=row['매출액(억)'],
            sales_increase_rate=row['전주 대비 증감률(%)_x'] if not pd.isna(row['전주 대비 증감률(%)_x']) else None,
            main_business_type=row['주 매출 업종'],
            visitors=row['방문인구(명)'],
            visitor_increase_rate=row['전주 대비 증감률(%)_y'] if not pd.isna(row['전주 대비 증감률(%)_y']) else None,
            main_age_group=row['주 방문 연령층'],
            main_time_period=row['주 방문 시간대']
        )
        session.add(festival_stat)
    
    # 성연령별_방문인구.csv 데이터 삽입
    for _, row in population_df.iterrows():
        population_stat = PopulationStats(
            region_id=region_id_map[int(row['영역 구분'])],
            gender=row['성별'],
            age_group=row['연령대'],
            period_type=row['구분'],
            visitors=row['방문인구(명)']
        )
        session.add(population_stat)
    
    # 업종별 매출.csv 데이터 삽입
    for _, row in sales_df.iterrows():
        sales_stat = SalesStats(
            region_id=region_id_map[int(row['영역 구분'])],
            business_type=row['업종명'],
            sales_amount=row['이용금액']
        )
        session.add(sales_stat)
    
    # 시간대별 인구 및 매출.csv 데이터 삽입
    for _, row in hourly_df.iterrows():
        hourly_stat = HourlyStats(
            region_id=region_id_map[int(row['영역 구분'])],
            period_type=row['구분'],
            time_period=row['시간대'],
            sales_amount=row['이용금액'],
            visitors=row['방문인구(명)']
        )
        session.add(hourly_stat)
    
    # 유입인구.csv 데이터 삽입
    for _, row in inflow_df.iterrows():
        inflow_stat = InflowStats(
            region_id=region_id_map[int(row['영역 구분'])],
            inflow_region_code=row['INFLOW_SGG_CD'],
            visitors=row['tot'],
            inflow_type=row['관내/관외'],
            region_code=row['SGG_CD'],
            region_name=row['SGG_NM'],
            province_name=row['SIDO_NM']
        )
        session.add(inflow_stat)
    
    # map_input_for_powerbi.csv 데이터 삽입
    for _, row in map_df.iterrows():
        map_data = MapData(
            center_lat=float(row['center_lat']) if not pd.isna(row['center_lat']) else None,
            center_lon=float(row['center_lon']) if not pd.isna(row['center_lon']) else None,
            total=float(row['tot']) if not pd.isna(row['tot']) else None,
            region_code=int(row['영역 구분']) if not pd.isna(row['영역 구분']) else None
        )
        session.add(map_data)
    
    # histo_for_powerbi.csv 데이터 삽입
    for _, row in histo_df.iterrows():
        histo_data = HistogramData(
            geometry=row['geometry'],
            total=row['tot'],
            region_code=int(row['영역 구분'])
        )
        session.add(histo_data)
    
    # tmap_fin.csv 데이터 삽입
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    for _, row in tmap_df.iterrows():
        tmap_data = TmapData(
            destination_name=str(row['frst_next_dstn_nm']),
            destination_x=safe_float(row['dstn_coord_x']),
            destination_y=safe_float(row['dstn_coord_y']),
            visit_count=int(row['vst_cnt']) if pd.notna(row['vst_cnt']) else None,
            stay_minutes=safe_float(row['sum_stay_min']),
            second_destination_addr=str(row['second_dstn_addr']),
            second_destination_x=safe_float(row['second_dstn_x']),
            second_destination_y=safe_float(row['second_dstn_y']),
            region_code=int(row['영역 구분']) if pd.notna(row['영역 구분']) else None,
            destination_type=str(row['목적지 구분'])
        )
        session.add(tmap_data)
    
    # 변경사항 커밋
    session.commit()
    session.close()
    
    print("데이터베이스 설정 및 데이터 로드가 완료되었습니다.")

if __name__ == "__main__":
    load_and_process_data()

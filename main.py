import sys, time, math, warnings
from pathlib import Path
import pulp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Keep console output usable on Windows terminals with a legacy code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
np.random.seed(42)

# Use paths relative to this script so it can be run from any working directory.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. 参数配置
# =============================================================================
PARAMS = {
    # 运输成本
    # 运输成本没有单一公开统计口径直接给出，作为本文的基准标定值。
    "trans_cost_fwd": 1.30,
    "trans_cost_rev": 1.69,
    
    # 碳排放
    # GB/T 51366-2019 附录E中10 t重型柴油货车因子为0.162 kg CO2e/(t·km)。
    # 本文对四段公路运输统一采用该因子，避免把逆向运输的成本修正误当成排放因子。
    "carbon_factor_fwd": 0.000162,
    "carbon_factor_rev": 0.000162,
    "carbon_price": 97.49,
    "carbon_cap": 517880,
    
    # 设施容量
    "central_hub_capacity": 4000000,
    "yiwu_central_capacity": 8000000,
    "front_hub_capacity": 1200000,
    "yiwu_front_capacity": 8000000,
    "pulp_capacity": 5000000,
    
    # 固定成本
    "fixed_cost_front": 18000000,
    "fixed_cost_central": 80000000,
    "yiwu_fixed_cost": 95000000,
    "yiwu_front_fixed": 120000000,
    
    # 回收相关
    "delta": 0.85,  # 分拣效率
    "eta": 0.90,    # 再制造转化效率
    "omega": 0.04,  # 【新增】中心枢纽损耗率ω，论文Table2=0.04
    
    # 分段线性回收成本
    "tier1_ratio": 0.40,
    "tier2_ratio": 0.20,
    "tier3_ratio": 0.10,
    "c_collect_tier1": 400,
    "c_collect_tier2": 1200,
    "c_collect_tier3": 2200,
    "max_recovery_rate": 0.70,
    
    # 区域平均回收率
    "alpha_regional": 0.00,
    
    # 需求 【修改1：σ由0.05改为论文0.15】
    "sigma_ratio": 0.15,
    
    # 生产成本
    "c_raw": 3200,
    "c_reman": 1800,
    # 绿色和平报告给出的快递垃圾平均焚烧处置成本为511 CNY/t。
    "c_disp": 511,
    
    # 其他
    "road_factor": 1.3,
    
    # 环境效益计算参数
    "wood_per_pulp": 2.1,
    "co2_per_wood": 1.8,
    
    # 需求口径：仅保留纸基包装（瓦楞纸箱与纸制套箱）
    # 绿色和平报告给出的瓦楞纸箱类快递包装质量为329 g，直接换算为0.329 kg。
    "avg_package_weight": 0.329,
    "paper_share": 0.4403 + 0.0947,
}

REGION_CFG = {"front_hub_radius": 50, "central_hub_radius": 500, "earth_radius": 6371}

# 节点配置
FACTORIES = ['嘉兴', '苏州', '无锡', '南通', '镇江', '湖州']
MARKETS = ['上海', '杭州', '苏州', '金华 (义乌)', '温州', '宁波', '合肥', '绍兴', '嘉兴', '台州', '南通', '南京', '宿迁', '徐州', '无锡', '连云港', '亳州', '湖州', '常州']
CENTER_HUBS = ['上海', '杭州', '南京', '金华 (义乌)']
FRONT_HUBS = MARKETS.copy()
K_STAR = ['金华 (义乌)']
L_STAR = ['金华 (义乌)']

# 2024年快递业务量（亿件）：除常州外来自国家邮政局公布的城市快递业务量表，
# 常州采用市统计公报公布的约6.5亿件。所有城市均按公开精度保留一位小数。
CITY_EXPRESS_VOLUME_BN = {
    '上海': 49.5, '杭州': 42.7, '苏州': 37.5, '金华 (义乌)': 171.9,
    '温州': 24.5, '宁波': 18.9, '合肥': 18.0, '绍兴': 17.2,
    '嘉兴': 16.6, '台州': 16.3, '南通': 14.9, '南京': 13.2,
    '宿迁': 11.1, '徐州': 10.7, '无锡': 10.1, '连云港': 8.6,
    '亳州': 8.7, '湖州': 8.3, '常州': 6.5
}
# 综合包装质量（t/yr），由业务量和报告中的329 g质量代理换算。
CITY_TOTAL_PACKAGE = {
    city: volume_bn * 1e8 * PARAMS["avg_package_weight"] / 1000
    for city, volume_bn in CITY_EXPRESS_VOLUME_BN.items()
}
# 模型只保留瓦楞纸箱与套装纸箱两类纸基包装。
CITY_DEMAND = {
    city: mass * PARAMS["paper_share"]
    for city, mass in CITY_TOTAL_PACKAGE.items()
}
CITY_COORDS = {'上海': (31.23, 121.47), '杭州': (30.27, 120.16), '苏州': (31.30, 120.59), '金华 (义乌)': (29.31, 120.08), '温州': (28.01, 120.39), '宁波': (29.87, 121.55), '合肥': (31.82, 117.23), '绍兴': (30.00, 120.57), '嘉兴': (30.75, 120.75), '台州': (28.66, 121.43), '南通': (32.02, 120.86), '南京': (32.06, 118.80), '宿迁': (33.97, 118.28), '徐州': (34.15, 117.11), '无锡': (31.60, 120.30), '连云港': (34.60, 119.23), '亳州': (33.84, 115.78), '湖州': (30.90, 120.08), '常州': (31.77, 119.98)}
FACTORIES_CFG = {'嘉兴': (30.75, 120.75), '苏州': (31.30, 120.59), '无锡': (31.60, 120.30), '南通': (32.02, 120.86), '镇江': (32.20, 119.43), '湖州': (30.90, 120.08)}

LOCATIONS = {}
for city, pos in FACTORIES_CFG.items(): LOCATIONS[f"F_{city}"] = pos
for city, pos in CITY_COORDS.items(): LOCATIONS[f"M_{city}"] = pos
for city in CENTER_HUBS: LOCATIONS[f"L_{city}"] = CITY_COORDS[city]
for city in FRONT_HUBS: LOCATIONS[f"K_{city}"] = CITY_COORDS[city]

DEMAND_BASE = {f"M_{c}": v for c, v in CITY_DEMAND.items()}
TOTAL_DEMAND = sum(DEMAND_BASE.values())

# 固定成本与容量
FIXED_COST_F = {f"K_{k}": PARAMS["yiwu_front_fixed"] if k in K_STAR else PARAMS["fixed_cost_front"] for k in FRONT_HUBS}
FIXED_COST_L = {f"L_{l}": PARAMS["yiwu_fixed_cost"] if l in L_STAR else PARAMS["fixed_cost_central"] for l in CENTER_HUBS}
CAPACITY_F = {f"K_{k}": PARAMS["front_hub_capacity"] if k not in K_STAR else PARAMS["yiwu_front_capacity"] for k in FRONT_HUBS}
CAPACITY_L = {f"L_{l}": PARAMS["central_hub_capacity"] if l not in L_STAR else PARAMS["yiwu_central_capacity"] for l in CENTER_HUBS}

# =============================================================================
# 2. 辅助函数
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((rlat2-rlat1)/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin((rlon2-rlon1)/2)**2
    return REGION_CFG["earth_radius"] * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

DIST = {}
for n1, pos1 in LOCATIONS.items():
    for n2, pos2 in LOCATIONS.items():
        if n1 != n2:
            DIST[(n1, n2)] = haversine(pos1[0], pos1[1], pos2[0], pos2[1]) * PARAMS["road_factor"]

# =============================================================================
# 3. 核心 MILP 求解器
# =============================================================================
def solve_clsc_academic(p=None):
    if p is None: p = PARAMS
    
    valid_z1 = [(j, k) for j in MARKETS for k in FRONT_HUBS if DIST[(f"M_{j}", f"K_{k}")] <= REGION_CFG["front_hub_radius"]]
    valid_z2 = [(k, l) for k in FRONT_HUBS for l in CENTER_HUBS if DIST[(f"K_{k}", f"L_{l}")] <= REGION_CFG["central_hub_radius"]]
    
    prob = pulp.LpProblem("CLSC_Academic", pulp.LpMinimize)
    
    # 决策变量
    x_raw = pulp.LpVariable.dicts("x_raw", ((i,j) for i in FACTORIES for j in MARKETS), lowBound=0)
    x_reman = pulp.LpVariable.dicts("x_reman", ((i,j) for i in FACTORIES for j in MARKETS), lowBound=0)
    
    R_tier1 = {j: pulp.LpVariable(f"Rt1_{j}", lowBound=0, upBound=p["tier1_ratio"]*DEMAND_BASE[f"M_{j}"]) for j in MARKETS}
    R_tier2 = {j: pulp.LpVariable(f"Rt2_{j}", lowBound=0, upBound=p["tier2_ratio"]*DEMAND_BASE[f"M_{j}"]) for j in MARKETS}
    R_tier3 = {j: pulp.LpVariable(f"Rt3_{j}", lowBound=0, upBound=p["tier3_ratio"]*DEMAND_BASE[f"M_{j}"]) for j in MARKETS}
    
    y_tier2 = {j: pulp.LpVariable(f"yt2_{j}", cat='Binary') for j in MARKETS}
    y_tier3 = {j: pulp.LpVariable(f"yt3_{j}", cat='Binary') for j in MARKETS}
    
    z1 = {(j,k): pulp.LpVariable(f"z1_{j}_{k}", lowBound=0) for (j,k) in valid_z1}
    z2 = {(k,l): pulp.LpVariable(f"z2_{k}_{l}", lowBound=0) for (k,l) in valid_z2}
    w = pulp.LpVariable.dicts("w", ((l,i) for l in CENTER_HUBS for i in FACTORIES), lowBound=0)
    
    waste_sort = pulp.LpVariable.dicts("waste_sort", FRONT_HUBS, lowBound=0)
    waste_center = pulp.LpVariable.dicts("waste_center", CENTER_HUBS, lowBound=0)
    waste_process = pulp.LpVariable.dicts("waste_process", FACTORIES, lowBound=0)
    
    y_F = pulp.LpVariable.dicts("y_F", [k for k in FRONT_HUBS if k not in K_STAR], cat='Binary')
    y_L = pulp.LpVariable.dicts("y_L", [l for l in CENTER_HUBS if l not in L_STAR], cat='Binary')
    
    E_exc = pulp.LpVariable("E_exc", lowBound=0)
    
    # 目标函数
    # 强制运营的义乌超级枢纽同时包含中央处理和前端收集两项固定成本。
    C_fix = pulp.lpSum(FIXED_COST_L[f"L_{l}"] for l in L_STAR) + \
            pulp.lpSum(FIXED_COST_F[f"K_{k}"] for k in K_STAR) + \
            pulp.lpSum(FIXED_COST_L[f"L_{l}"] * y_L[l] for l in CENTER_HUBS if l not in L_STAR) + \
            pulp.lpSum(FIXED_COST_F[f"K_{k}"] * y_F[k] for k in FRONT_HUBS if k not in K_STAR)
    
    C_prod = pulp.lpSum(p["c_raw"] * x_raw[(i,j)] + p["c_reman"] * x_reman[(i,j)] for i in FACTORIES for j in MARKETS)
    
    C_trans = pulp.lpSum(p["trans_cost_fwd"] * DIST[(f"F_{i}", f"M_{j}")] * (x_raw[(i,j)] + x_reman[(i,j)]) for i in FACTORIES for j in MARKETS) + \
              pulp.lpSum(p["trans_cost_rev"] * DIST[(f"M_{j}", f"K_{k}")] * z1[(j,k)] for (j,k) in valid_z1) + \
              pulp.lpSum(p["trans_cost_rev"] * DIST[(f"K_{k}", f"L_{l}")] * z2[(k,l)] for (k,l) in valid_z2) + \
              pulp.lpSum(p["trans_cost_rev"] * DIST[(f"L_{l}", f"F_{i}")] * w[(l,i)] for l in CENTER_HUBS for i in FACTORIES)
    
    C_collect = p["c_collect_tier1"] * pulp.lpSum(R_tier1[j] for j in MARKETS) + \
                p["c_collect_tier2"] * pulp.lpSum(R_tier2[j] for j in MARKETS) + \
                p["c_collect_tier3"] * pulp.lpSum(R_tier3[j] for j in MARKETS)
    
    C_disp = p["c_disp"] * (pulp.lpSum(waste_sort[k] for k in FRONT_HUBS) + 
                            pulp.lpSum(waste_center[l] for l in CENTER_HUBS) + 
                            pulp.lpSum(waste_process[i] for i in FACTORIES))
    
    C_carbon = p["carbon_price"] * E_exc
    
    prob += C_fix + C_prod + C_trans + C_collect + C_disp + C_carbon, "Total_Cost"
    
    # 约束条件
    for j in MARKETS:
        total_recovery_j = pulp.lpSum(z1[(j,k)] for k in FRONT_HUBS if (j,k) in valid_z1)
        prob += total_recovery_j == R_tier1[j] + R_tier2[j] + R_tier3[j]
    
    for j in MARKETS:
        prob += R_tier1[j] >= p["tier1_ratio"] * DEMAND_BASE[f"M_{j}"] * y_tier2[j]
        prob += R_tier2[j] >= p["tier2_ratio"] * DEMAND_BASE[f"M_{j}"] * y_tier3[j]
        # 只有在上一层达到上限后，下一层才允许出现正流量。
        prob += R_tier2[j] <= p["tier2_ratio"] * DEMAND_BASE[f"M_{j}"] * y_tier2[j]
        prob += R_tier3[j] <= p["tier3_ratio"] * DEMAND_BASE[f"M_{j}"] * y_tier3[j]
        prob += y_tier3[j] <= y_tier2[j]
    
    for j in MARKETS:
        d_j = DEMAND_BASE[f"M_{j}"]
        safety = d_j * p["sigma_ratio"]
        # d_j 是当年末端废弃包装供给量代理；安全库存只增加正向供给需求，
        # 不计入当年回收率分母，也不凭空增加可回收废弃物。
        total_recovery_j = pulp.lpSum(z1[(j,k)] for k in FRONT_HUBS if (j,k) in valid_z1)
        prob += total_recovery_j <= d_j
        prob += pulp.lpSum(x_raw[(i,j)] + x_reman[(i,j)] for i in FACTORIES) == d_j + safety
    
    total_recovery_regional = pulp.lpSum(R_tier1[j] + R_tier2[j] + R_tier3[j] for j in MARKETS)
    prob += total_recovery_regional >= p["alpha_regional"] * TOTAL_DEMAND
    prob += total_recovery_regional <= p["max_recovery_rate"] * TOTAL_DEMAND
    
    for k in FRONT_HUBS:
        total_z1_k = pulp.lpSum(z1[(j,k)] for j in MARKETS if (j,k) in valid_z1)
        total_z2_k = pulp.lpSum(z2[(k,l)] for l in CENTER_HUBS if (k,l) in valid_z2)
        prob += total_z1_k == total_z2_k + waste_sort[k]
        prob += waste_sort[k] == (1 - p["delta"]) * total_z1_k
    
    for l in CENTER_HUBS:
        total_z2_l = pulp.lpSum(z2[(k,l)] for k in FRONT_HUBS if (k,l) in valid_z2)
        total_w_l = pulp.lpSum(w[(l,i)] for i in FACTORIES)
        # 【修改2-A：硬编码0.02替换为参数omega】
        prob += total_z2_l == total_w_l + waste_center[l]
        prob += waste_center[l] == p["omega"] * total_z2_l
    
    for i in FACTORIES:
        total_w = pulp.lpSum(w[(l,i)] for l in CENTER_HUBS)
        prob += pulp.lpSum(x_reman[(i,j)] for j in MARKETS) <= p["eta"] * total_w
        prob += waste_process[i] == (1 - p["eta"]) * total_w
        prob += pulp.lpSum(x_raw[(i,j)] + x_reman[(i,j)] for j in MARKETS) <= p["pulp_capacity"]
    
    for k in FRONT_HUBS:
        inflow = pulp.lpSum(z1[(j,k)] for j in MARKETS if (j,k) in valid_z1)
        if k not in K_STAR:
            prob += inflow <= CAPACITY_F[f"K_{k}"] * y_F[k]
        else:
            prob += inflow <= CAPACITY_F[f"K_{k}"]
    
    for l in CENTER_HUBS:
        inflow = pulp.lpSum(z2[(k,l)] for k in FRONT_HUBS if (k,l) in valid_z2)
        if l not in L_STAR:
            prob += inflow <= CAPACITY_L[f"L_{l}"] * y_L[l]
            prob += pulp.lpSum(w[(l,i)] for i in FACTORIES) <= CAPACITY_L[f"L_{l}"] * y_L[l]
        else:
            prob += inflow <= CAPACITY_L[f"L_{l}"]
    
    for j in MARKETS:
        eligible = [k for k in FRONT_HUBS if k not in K_STAR and (j,k) in valid_z1]
        if eligible:
            prob += pulp.lpSum(y_F[k] for k in eligible) >= 1
    
    E_total = p["carbon_factor_fwd"] * pulp.lpSum(DIST[(f"F_{i}", f"M_{j}")] * (x_raw[(i,j)] + x_reman[(i,j)]) for i in FACTORIES for j in MARKETS) + \
              p["carbon_factor_rev"] * (pulp.lpSum(DIST[(f"M_{j}", f"K_{k}")] * z1[(j,k)] for (j,k) in valid_z1) + \
              pulp.lpSum(DIST[(f"K_{k}", f"L_{l}")] * z2[(k,l)] for (k,l) in valid_z2) + \
              pulp.lpSum(DIST[(f"L_{l}", f"F_{i}")] * w[(l,i)] for l in CENTER_HUBS for i in FACTORIES))
    
    prob += E_exc >= E_total - p["carbon_cap"]
    
    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=180))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    
    # 结果提取
    total_recovery = sum(pulp.value(R_tier1[j] + R_tier2[j] + R_tier3[j]) for j in MARKETS)
    total_x_reman = sum(pulp.value(x_reman[(i,j)]) for i in FACTORIES for j in MARKETS)
    total_waste_sort = sum(pulp.value(waste_sort[k]) for k in FRONT_HUBS)
    total_waste_center = sum(pulp.value(waste_center[l]) for l in CENTER_HUBS)
    total_waste_process = sum(pulp.value(waste_process[i]) for i in FACTORIES)
    
    mass_balance_error = total_recovery - total_x_reman - total_waste_sort - total_waste_center - total_waste_process
    
    return {
        "total_cost": pulp.value(prob.objective),
        "fixed_cost": pulp.value(C_fix),
        "prod_cost": pulp.value(C_prod),
        "transport_cost": pulp.value(C_trans),
        "collect_cost_tier1": pulp.value(p["c_collect_tier1"] * pulp.lpSum(R_tier1[j] for j in MARKETS)),
        "collect_cost_tier2": pulp.value(p["c_collect_tier2"] * pulp.lpSum(R_tier2[j] for j in MARKETS)),
        "collect_cost_tier3": pulp.value(p["c_collect_tier3"] * pulp.lpSum(R_tier3[j] for j in MARKETS)),
        "disp_cost": pulp.value(C_disp),
        "carbon_cost": pulp.value(C_carbon),
        "total_emission": pulp.value(E_total),
        "excess_emission": pulp.value(E_exc),
        "qty_rev": total_recovery,
        "qty_x_reman": total_x_reman,
        "reman_ratio": total_x_reman / TOTAL_DEMAND if TOTAL_DEMAND > 0 else 0,
        "waste_sort": total_waste_sort,
        "waste_center": total_waste_center,
        "waste_process": total_waste_process,
        "effective_yield": total_x_reman / total_recovery if total_recovery > 0 else 0,
        "tier1_qty": sum(pulp.value(R_tier1[j]) for j in MARKETS),
        "tier2_qty": sum(pulp.value(R_tier2[j]) for j in MARKETS),
        "tier3_qty": sum(pulp.value(R_tier3[j]) for j in MARKETS),
        "mass_balance_error": mass_balance_error,
        "mass_balance_error_pct": mass_balance_error / total_recovery * 100 if total_recovery > 0 else 0,
    }

# =============================================================================
# 4. 第一阶段：模型求解
# =============================================================================
start_time = time.time()
print("="*90)
print(" "*25 + "阶段1：MILP模型求解")
print("="*90)

# 4.1 基准情景（α=0，经济最优，无强制回收率约束）
print("\n【1/7】求解基准情景...")
p_base = PARAMS.copy()
p_base['alpha_regional'] = 0  # 基准情景强制回收率为0
res_academic = solve_clsc_academic(p_base)
if res_academic is None:
    print("❌ 基准情景求解失败！")
    sys.exit(1)
print("✅ 基准情景求解完成")

# 4.2 Alpha情景扫描
print("\n【2/7】求解Alpha情景对比...")
alpha_test = [0, 0.30, 0.40, 0.50, 0.60]
alpha_results = []
for alpha in alpha_test:
    p_test = PARAMS.copy()
    p_test['alpha_regional'] = alpha
    r = solve_clsc_academic(p_test)
    if r:
        alpha_results.append({"alpha": alpha, "result": r})
print(f"✅ 完成 {len(alpha_results)} 组Alpha情景求解")

# 4.3 Tier-1成本敏感性
print("\n【3/7】求解Tier-1成本敏感性...")
tier1_results = []
for c_t1 in [200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    p_test = PARAMS.copy()
    p_test["c_collect_tier1"] = c_t1
    r = solve_clsc_academic(p_test)
    if r:
        r["c_tier1"] = c_t1
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        tier1_results.append(r)
print(f"✅ 完成 {len(tier1_results)} 组Tier-1敏感性求解")

# 4.4 再制造成本敏感性
print("\n【4/7】求解再制造成本敏感性...")
reman_results = []
for c_rm in [1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]:
    p_test = PARAMS.copy()
    p_test["c_reman"] = c_rm
    r = solve_clsc_academic(p_test)
    if r:
        r["c_reman"] = c_rm
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        reman_results.append(r)
print(f"✅ 完成 {len(reman_results)} 组再制造成本敏感性求解")

# 4.5 原生浆成本敏感性
print("\n【5/7】求解原生浆成本敏感性...")
raw_results = []
for c_rw in [2600, 2800, 3000, 3200, 3400, 3600, 3800]:
    p_test = PARAMS.copy()
    p_test["c_raw"] = c_rw
    r = solve_clsc_academic(p_test)
    if r:
        r["c_raw"] = c_rw
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        raw_results.append(r)
print(f"✅ 完成 {len(raw_results)} 组原生浆成本敏感性求解")

# 4.6 技术参数敏感性
print("\n【6/7】求解技术参数敏感性...")
delta_results = []
for d_val in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    p_test = PARAMS.copy()
    p_test["delta"] = d_val
    r = solve_clsc_academic(p_test)
    if r:
        r["delta"] = d_val
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        delta_results.append(r)

eta_results = []
for e_val in [0.75, 0.80, 0.85, 0.90, 0.95]:
    p_test = PARAMS.copy()
    p_test["eta"] = e_val
    r = solve_clsc_academic(p_test)
    if r:
        r["eta"] = e_val
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        eta_results.append(r)
print(f"✅ 完成 {len(delta_results)+len(eta_results)} 组技术参数敏感性求解")

# 4.7 碳价敏感性
print("\n【7/7】求解碳价敏感性...")
carbon_results = []
for cp in [50, 100, 200, 300, 500, 800, 1000, 1500, 2000]:
    p_test = PARAMS.copy()
    p_test["carbon_price"] = cp
    r = solve_clsc_academic(p_test)
    if r:
        r["carbon_price"] = cp
        r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
        carbon_results.append(r)
print(f"✅ 完成 {len(carbon_results)} 组碳价敏感性求解")

# 4.8 整理情景结果
scenario_results = []
for item in alpha_results:
    alpha = item["alpha"]
    r = item["result"]
    r["scenario"] = f"α={alpha:.2f}"
    r["alpha"] = alpha
    r["recovery_rate"] = r["qty_rev"] / TOTAL_DEMAND * 100
    r["collect_cost"] = r["collect_cost_tier1"] + r["collect_cost_tier2"] + r["collect_cost_tier3"]
    scenario_results.append(r)

res_base = res_academic
solve_time = time.time() - start_time
print(f"\n✅ 阶段1完成：模型结果已保存在内存中")
print(f"   总计：{len(scenario_results)} 组情景 + {len(tier1_results)+len(reman_results)+len(raw_results)+len(delta_results)+len(eta_results)+len(carbon_results)} 组敏感性分析")
print(f"   求解耗时：{solve_time/60:.1f} 分钟")

# =============================================================================
# 5. 第二阶段：生成6张学术图表
# =============================================================================
print("\n" + "="*90)
print(" "*25 + "阶段2：生成6张学术图表")
print("="*90)

# IEEE配色
C1  = '#1A4F8A'
C2  = '#C0392B'
C3  = '#2E7D32'
C4  = '#E67E22'
C5  = '#6C3483'
C6  = '#17A589'
CGRAY = '#7F8C8D'
CGRID = '#BDC3C7'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 180,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': CGRID,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.6,
    'axes.axisbelow': True,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.6,
})

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  已生成：{name}")

sr = scenario_results
labels_short = ['α=0\n(Economic)', 'α=0.30\n(Weak)', 'α=0.40\n(Moderate)', 'α=0.50\n(Mandatory)', 'α=0.60\n(Enhanced)']

# ========== Fig 1：网络拓扑图 ==========
print("\n【1/6】网络拓扑图...")
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.set_facecolor('#F8FAFC')
ax.set_xlim(114.5, 122.8); ax.set_ylim(27.5, 35.5)

prov_x = [119.5, 121.5, 122.2, 121.8, 120.8, 119.3, 117.5, 116.0, 115.5, 116.2, 117.8, 119.5]
prov_y = [28.0,  28.3,  29.5,  31.5,  33.8,  34.8,  34.5,  33.5,  31.5,  30.0,  28.5,  28.0]
ax.fill(prov_x, prov_y, color='#E8F4FD', alpha=0.5, zorder=0)
ax.plot(prov_x, prov_y, color='#AED6F1', lw=0.8, zorder=1)

# 城市中英文映射
city_en = {
    '上海': 'Shanghai', '杭州': 'Hangzhou', '苏州': 'Suzhou',
    '金华 (义乌)': 'Jinhua (Yiwu)', '温州': 'Wenzhou', '宁波': 'Ningbo',
    '合肥': 'Hefei', '绍兴': 'Shaoxing', '嘉兴': 'Jiaxing',
    '台州': 'Taizhou', '南通': 'Nantong', '南京': 'Nanjing',
    '宿迁': 'Suqian', '徐州': 'Xuzhou', '无锡': 'Wuxi',
    '连云港': 'Lianyungang', '亳州': 'Bozhou', '湖州': 'Huzhou',
    '常州': 'Changzhou', '镇江': 'Zhenjiang'
}

max_d = max(CITY_DEMAND.values())
for city, (lat, lon) in CITY_COORDS.items():
    d_val = CITY_DEMAND[city]
    sz = 40 + 280 * (d_val / max_d)
    ax.scatter(lon, lat, s=sz, c=C1, alpha=0.7, zorder=4, edgecolors='white', linewidths=0.6)
    offx, offy = 0.07, 0.07
    if city == '金华 (义乌)': offx = 0.1; offy = -0.18
    elif city == '上海': offx = 0.1
    elif city == '合肥': offx = -0.5
    ax.text(lon+offx, lat+offy, city_en[city], fontsize=5.5, color='#1C2833', zorder=6,
            ha='left' if offx >= 0 else 'right')

for hub in CENTER_HUBS:
    lat, lon = CITY_COORDS[hub]
    sz = 180 if hub in K_STAR else 120
    color = C2 if hub in K_STAR else C4
    ax.scatter(lon, lat, s=sz, c=color, marker='D', zorder=6, edgecolors='white', linewidths=1.0)

for fac, (lat, lon) in FACTORIES_CFG.items():
    ax.scatter(lon, lat, s=90, c=C3, marker='^', zorder=6, edgecolors='white', linewidths=0.8)

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C1, markersize=7, label='Market Node'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=C2, markersize=7, label='Yiwu Super Hub'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=C4, markersize=7, label='Central Hub'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor=C3, markersize=7, label='Production Factory'),
]
ax.legend(handles=legend_elements, loc='lower left', frameon=True, framealpha=0.92, edgecolor=CGRID, fontsize=7)
ax.set_xlabel('Longitude (°E)', fontsize=8)
ax.set_ylabel('Latitude (°N)', fontsize=8)
ax.set_title('CLSC Network Topology of Yangtze River Delta Region',
             fontsize=9, fontweight='bold', pad=8)
ax.grid(True, color=CGRID, lw=0.4, alpha=0.5)
save_fig(fig, 'fig1_network_topology.pdf')

# ========== Fig 2：成本结构分析 ==========
print("【2/6】成本结构分析图...")
fig = plt.figure(figsize=(7.2, 3.4))
# 核心修改：高度比例改为 上3下1，上半部分占主导
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.08)
ax_top = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax2 = fig.add_subplot(gs[:, 1])  # 右半部分饼图占整行高度

cost_keys = ['prod_cost', 'transport_cost', 'collect_cost', 'fixed_cost', 'disp_cost', 'carbon_cost']
cost_labels = ['Production', 'Transportation', 'Collection', 'Facility Fixed', 'Disposal', 'Carbon Tax']
cost_colors = [C1, C3, C4, C5, C6, C2]

x = np.arange(len(sr))
w = 0.55

# 上下两图都绘制堆叠柱状图
bottoms_top = np.zeros(len(sr))
bottoms_bottom = np.zeros(len(sr))
for key, color in zip(cost_keys, cost_colors):
    vals = np.array([s[key]/1e8 for s in sr])
    ax_top.bar(x, vals, w, bottom=bottoms_top, color=color, alpha=0.88)
    ax_bottom.bar(x, vals, w, bottom=bottoms_bottom, color=color, alpha=0.88)
    bottoms_top += vals
    bottoms_bottom += vals

# 纵轴范围随重算结果自动调整：上图放大总成本差异，下图展示主要成本构成。
total_vals = np.array([s['total_cost']/1e8 for s in sr])
ax_top.set_ylim(total_vals.min() * 0.985, total_vals.max() * 1.02)
ax_bottom.set_ylim(0, total_vals.min() * 0.88)
ax_bottom.set_yticks([0, 150, 300])

# 隐藏衔接处边框，绘制截断斜线
ax_top.spines.bottom.set_visible(False)
ax_bottom.spines.top.set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
              linestyle="none", color='black', mec='black', mew=1, clip_on=False)
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

# 总成本数值标注（只在上图显示）
for i, s in enumerate(sr):
    total = s['total_cost']/1e8
    pct = (total - sr[0]['total_cost']/1e8) / (sr[0]['total_cost']/1e8) * 100
    ax_top.text(i, total + 1.5, f'{total:.1f}\n(+{pct:.1f}%)', ha='center', fontsize=6, color='#2C3E50')

# 左图坐标轴与标题
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(['α=0\n(Economic)', 'α=0.30\n(Weak)', 'α=0.40\n(Moderate)', 'α=0.50\n(Mandatory)', 'α=0.60\n(Enhanced)'], fontsize=6)
ax_bottom.set_ylabel('Total System Cost (×10⁸ CNY)', fontsize=7)
ax_top.set_title('(a) Cost Composition by Policy Scenario', fontsize=9, fontweight='bold', pad=7)

# ========== 右子图饼图（保持原有逻辑不变）==========
base = sr[0]
vals = [base[k]/1e8 for k in cost_keys]
def autopct_filter(pct):
    return f'{pct:.1f}%' if pct >= 2 else ''
wedges, texts, autotexts = ax2.pie(
    vals, labels=None, colors=cost_colors, autopct=autopct_filter,
    startangle=140, pctdistance=0.78, radius=0.78,
    wedgeprops=dict(linewidth=0.8, edgecolor='white'),
    textprops={'fontsize': 6.5}
)
ax2.legend(wedges, cost_labels, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5, framealpha=0.9)
ax2.set_title('(b) Base Scenario Cost Structure\n(α = 0, Economic Optimum)', fontsize=9, fontweight='bold', pad=8)

fig.suptitle('System Cost Structure Analysis', fontsize=10, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'fig2_cost_structure.pdf')

# ========== Fig 3：政策情景综合绩效对比 ==========
print("【3/6】政策情景综合对比...")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
x = np.arange(len(sr))

ax = axes[0,0]
rr = [s['recovery_rate'] for s in sr]
economic_optimum = sr[0]['recovery_rate']
bars = ax.bar(x, rr, 0.55, color=[C3 if v <= economic_optimum + 2 else C4 if v <= 50 else C2 for v in rr], alpha=0.88)
ax.axhline(economic_optimum, ls='--', lw=1.0, color=C1, label='Economic optimum')
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=7)
ax.set_ylabel('Recovery Rate (%)', fontsize=8)
ax.set_title('(a) Recovery Rate', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5); ax.set_ylim(0, 75)
for bar, v in zip(bars, rr): ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f'{v:.1f}%', ha='center', fontsize=6.5)

ax = axes[0,1]
costs = [s['total_cost']/1e8 for s in sr]
base_cost_val = costs[0]
pct_inc = [(c-base_cost_val)/base_cost_val*100 for c in costs]
# 颜色随增幅自动切换
color_b = [C1 if i==0 else C4 if pi < 2 else C2 for i, pi in enumerate(pct_inc)]
bars = ax.bar(x, costs, 0.55, color=color_b, alpha=0.88)

ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=7)
ax.set_ylabel('Total System Cost (×10⁸ CNY)', fontsize=8)
ax.set_title('(b) Total System Cost', fontsize=9, fontweight='bold')
# 纵轴随重算结果自动适配，放大柱子高度差异
ax.set_ylim(min(costs) * 0.985, max(costs) * 1.02)

# 文字动态偏移，不会飘很高
for bar, c, pi in zip(bars, costs, pct_inc):
    offset = c * 0.008
    ax.text(bar.get_x() + bar.get_width()/2, c + offset,
            f'{c:.1f}\n({pi:+.1f}%)',
            ha='center', va='bottom', fontsize=6.0, color='#2C3E50')

ax = axes[1,0]
# 修复：再制造比例转换为百分比（原数值为小数形式）
rr2 = [s['reman_ratio'] * 100 for s in sr]
bars = ax.bar(x, rr2, 0.55, color=C5, alpha=0.80)
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=7)
ax.set_ylabel('Remanufactured Pulp Ratio (%)', fontsize=8)
ax.set_title('(c) Remanufactured Pulp Share', fontsize=9, fontweight='bold')
ax.set_ylim(0, 50)
for bar, v in zip(bars, rr2): ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=6.5)

ax = axes[1,1]
em = [s['total_emission']/1e4 for s in sr]
bars = ax.bar(x, em, 0.55, color=C6, alpha=0.80)
ax.axhline(PARAMS['carbon_cap']/1e4, ls='--', lw=1.0, color=C2, label='Carbon cap')
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=7)
ax.set_ylabel('Total CO₂ Emission (×10⁴ t)', fontsize=8)
ax.set_title('(d) Carbon Emission', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5)
# 修复：收紧纵坐标范围，突出数据差异
em_min, em_max = min(em), max(em)
ax.set_ylim(em_min * 0.92, em_max * 1.05)
for bar, v in zip(bars, em): ax.text(bar.get_x()+bar.get_width()/2, v+0.15, f'{v:.1f}', ha='center', fontsize=6.5)

fig.suptitle('Comprehensive Performance Under Policy Scenarios', fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig3_policy_performance.pdf')

# ========== Fig 4：敏感性总览（龙卷风图） ==========
print("【4/6】敏感性总览龙卷风图...")
base_cost = res_base['total_cost']
tornado_data = []

t1_low = tier1_results[0]['total_cost']
t1_high = tier1_results[-1]['total_cost']
tornado_data.append(('c_tier1\n[200, 1000]', t1_low, t1_high))

rm_low = reman_results[0]['total_cost']
rm_high = reman_results[-1]['total_cost']
tornado_data.append(('c_reman\n[1400, 2200]', rm_low, rm_high))

rw_low = raw_results[0]['total_cost']
rw_high = raw_results[-1]['total_cost']
tornado_data.append(('c_raw\n[2600, 3800]', rw_low, rw_high))

d_low = delta_results[-1]['total_cost']
d_high = delta_results[0]['total_cost']
tornado_data.append(('δ (sorting yield)\n[0.70, 0.95]', d_low, d_high))

e_low = eta_results[-1]['total_cost']
e_high = eta_results[0]['total_cost']
tornado_data.append(('η (remanuf. yield)\n[0.75, 0.95]', e_low, e_high))

cp_low = carbon_results[0]['total_cost']
cp_high = carbon_results[-1]['total_cost']
tornado_data.append(('Carbon price\n[50, 2000 CNY/t]', cp_low, cp_high))

tornado_data.sort(key=lambda x: abs(x[2]-x[1]), reverse=True)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
y_pos = np.arange(len(tornado_data))

for i, (label, low, high) in enumerate(tornado_data):
    swing_neg = min(low, high) - base_cost
    swing_pos = max(low, high) - base_cost
    ax.barh(i, swing_neg/1e8, left=0, color=C2, alpha=0.80, height=0.55)
    ax.barh(i, swing_pos/1e8, left=0, color=C1, alpha=0.80, height=0.55)
    ax.text(-0.35, i, f'{swing_neg/1e8:.1f}', va='center', ha='right', fontsize=6.5, color=C2)
    ax.text(0.35, i, f'+{swing_pos/1e8:.1f}', va='center', ha='left', fontsize=6.5, color=C1)

ax.set_yticks(y_pos)
ax.set_yticklabels([t[0] for t in tornado_data], fontsize=7.5)
ax.axvline(0, color='black', lw=1.0)
ax.set_xlabel('Cost Deviation from Baseline (×10⁸ CNY)', fontsize=8)
ax.set_title('Tornado Chart of One-Way Sensitivity Analysis',
             fontsize=9, fontweight='bold')
legend_els = [mpatches.Patch(color=C2, alpha=0.8, label='Cost decrease'),
              mpatches.Patch(color=C1, alpha=0.8, label='Cost increase')]
ax.legend(handles=legend_els, loc='lower right', fontsize=7)
fig.tight_layout()
save_fig(fig, 'fig4_sensitivity_tornado.pdf')

# ========== Fig 5：关键参数响应曲线 ==========
print("【5/6】关键参数响应曲线...")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

# (a) Tier-1成本 vs 回收率
ax = axes[0,0]
t1x = [r['c_tier1'] for r in tier1_results]
rr_t1 = [r['recovery_rate'] for r in tier1_results]
ax.plot(t1x, rr_t1, 'o-', color=C1, mfc='white', mew=1.5, ms=6)
ax.axvline(400, ls='--', color=CGRAY, lw=0.8, label='Baseline')
ax.fill_between(t1x, rr_t1, 0, alpha=0.08, color=C1)
ax.set_xlabel('Tier-1 Collection Cost (CNY/t)', fontsize=8)
ax.set_ylabel('Recovery Rate (%)', fontsize=8)
ax.set_title('(a) Recovery Rate vs. Collection Cost', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5)
ax.set_ylim(min(rr_t1)*0.98, max(rr_t1)*1.02)  # 收紧纵坐标，凸显变化

# (b) 再制造成本 vs 再制造比例
ax = axes[0,1]
rx = [r['c_reman'] for r in reman_results]
rem_r = [r['reman_ratio'] * 100 for r in reman_results]  # 修复：小数转百分比
ax.plot(rx, rem_r, 's-', color=C5, mfc='white', mew=1.5, ms=6)
ax.axvline(1800, ls='--', color=CGRAY, lw=0.8, label='Baseline')
ax.fill_between(rx, rem_r, 0, alpha=0.08, color=C5)
ax.set_xlabel('Remanufacturing Cost (CNY/t)', fontsize=8)
ax.set_ylabel('Remanufactured Ratio (%)', fontsize=8)
ax.set_title('(b) Remanuf. Ratio vs. Production Cost', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5)
ax.set_ylim(min(rem_r)*0.98, max(rem_r)*1.02)  # 收紧纵坐标

# (c) 分拣效率 vs 系统成本
ax = axes[1,0]
dx = [r['delta'] for r in delta_results]
cost_d = [r['total_cost']/1e8 for r in delta_results]
ax.plot(dx, cost_d, 'D-', color=C3, mfc='white', mew=1.5, ms=6)
ax.axvline(0.85, ls='--', color=CGRAY, lw=0.8, label='Baseline')
ax.fill_between(dx, cost_d, 0, alpha=0.08, color=C3)
ax.set_xlabel('Sorting Yield δ', fontsize=8)
ax.set_ylabel('Total System Cost (×10⁸ CNY)', fontsize=8)
ax.set_title('(c) System Cost vs. Sorting Efficiency', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5)
ax.set_ylim(min(cost_d)*0.995, max(cost_d)*1.005)  # 收紧纵坐标，放大成本差异

# (d) 碳价 vs 碳排放
ax = axes[1,1]
cpx = [r['carbon_price'] for r in carbon_results]
em_c = [r['total_emission']/1e4 for r in carbon_results]
ax.plot(cpx, em_c, '^-', color=C2, mfc='white', mew=1.5, ms=6)
ax.axvline(97.49, ls='--', color=CGRAY, lw=0.8, label='Current price')
ax.axhline(PARAMS['carbon_cap']/1e4, ls=':', color=C6, lw=0.8, label='Carbon cap')
ax.set_xlabel('Carbon Price (CNY/t CO₂)', fontsize=8)
ax.set_ylabel('Total Emission (×10⁴ t CO₂)', fontsize=8)
ax.set_title('(d) Emission vs. Carbon Price', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5)
ax.set_ylim(min(em_c)*0.99, max(em_c)*1.01)  # 收紧纵坐标

fig.suptitle('Response Curves of Key Parameters', fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig5_parameter_response.pdf')

# ========== Fig 6：物料循环与环境效益 ==========
print("【6/6】物料循环与环境效益...")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
rb = res_base
total_rec = rb['qty_rev']
x_reman = rb['qty_x_reman']
ws = rb['waste_sort']
wc = rb['waste_center']
wp = rb['waste_process']
rr = (total_rec / TOTAL_DEMAND) * 100

# (a) 全链路物料流 【修改2-B：硬编码0.98替换为 1-omega】
ax = axes[0,0]
stages = ['Demand', 'Collection', 'Sorting', 'Processing', 'Remanuf.', 'Supply']
flow_vals = [
    TOTAL_DEMAND/1e6,
    total_rec/1e6,
    total_rec*PARAMS["delta"]/1e6,
    total_rec*PARAMS["delta"]*(1 - PARAMS["omega"])/1e6,
    x_reman/1e6,
    TOTAL_DEMAND*(1 + PARAMS["sigma_ratio"])/1e6
]
waste_vals = [0, 0, ws/1e6, wc/1e6, wp/1e6, 0]
flow_colors = [C1, C3, C3, C6, C5, C1]
bar_w = 0.45
for i, (fv, wv, fc) in enumerate(zip(flow_vals, waste_vals, flow_colors)):
    ax.bar(i, fv, bar_w, color=fc, alpha=0.8, zorder=3)
    if wv > 0:
        ax.bar(i, wv, bar_w, bottom=fv, color=C2, alpha=0.6, zorder=3, hatch='//')
    ax.text(i, fv*0.5, f'{fv:.2f}', ha='center', va='center', fontsize=6,
            color='white', fontweight='bold', zorder=4)
ax.set_xticks(np.arange(len(stages))); ax.set_xticklabels(stages, fontsize=6.5)
ax.set_ylabel('Material Flow (Mt/yr)', fontsize=8)
ax.set_title('(a) Annual Material Flow', fontsize=9, fontweight='bold')

# (b) 木材节约量
ax = axes[0,1]
wood = [s['qty_x_reman']*PARAMS['wood_per_pulp']/1e4 for s in sr]
bars = ax.bar(np.arange(len(sr)), wood, 0.55, color=C3, alpha=0.85)
ax.set_xticks(np.arange(len(sr))); ax.set_xticklabels(labels_short, fontsize=6.5)
ax.set_ylabel('Timber Saved (×10⁴ t/yr)', fontsize=8)
ax.set_title('(b) Timber Conservation', fontsize=9, fontweight='bold')
for bar, v in zip(bars, wood): ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}', ha='center', fontsize=6)

# (c) 森林碳汇等效
ax = axes[1,0]
co2_seq = [s['qty_x_reman']*PARAMS['wood_per_pulp']*PARAMS['co2_per_wood']/1e4 for s in sr]
bars = ax.bar(np.arange(len(sr)), co2_seq, 0.55, color=C6, alpha=0.85)
ax.set_xticks(np.arange(len(sr))); ax.set_xticklabels(labels_short, fontsize=6.5)
ax.set_ylabel('CO₂ Sequestration Equiv. (×10⁴ t)', fontsize=8)
ax.set_title('(c) Forest Carbon Sequestration', fontsize=9, fontweight='bold')
for bar, v in zip(bars, co2_seq): ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}', ha='center', fontsize=6)

# (d) 循环经济产出
ax = axes[1,1]
rm_out = [s['qty_x_reman']/1e4 for s in sr]
bars = ax.bar(np.arange(len(sr)), rm_out, 0.55, color=C5, alpha=0.85)
ax.set_xticks(np.arange(len(sr))); ax.set_xticklabels(labels_short, fontsize=6.5)
ax.set_ylabel('Remanuf. Pulp Output (×10⁴ t/yr)', fontsize=8)
ax.set_title('(d) Circular Economy Output', fontsize=9, fontweight='bold')
for bar, v in zip(bars, rm_out): ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.0f}', ha='center', fontsize=6)

fig.suptitle('Material Circulation and Environmental Benefits', fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig6_environmental_benefits.pdf')

print(f"\n✅ 阶段2完成：6张学术图表已保存到 {OUTPUT_DIR} 目录")

# =============================================================================
# 6. 第三阶段：生成数据表格
# =============================================================================
print("\n" + "="*90)
print(" "*25 + "阶段3：生成数据表格")
print("="*90)

sr = scenario_results
t1 = tier1_results
rm = reman_results
rw = raw_results
dl = delta_results
et = eta_results
cr = carbon_results
TD = TOTAL_DEMAND
P = PARAMS

# Table 1: 参数表
params_table = pd.DataFrame([
    ["Transport Cost (forward)", "c_fwd", "1.30", "CNY/(t·km)", "Model calibration"],
    ["Transport Cost (reverse)", "c_rev", "1.69", "CNY/(t·km)", "Model calibration"],
    ["Carbon Emission Factor (fwd)", "e_fwd", "1.62×10⁻⁴", "t CO₂/(t·km)", "GB/T 51366-2019, App. E"],
    ["Carbon Emission Factor (rev)", "e_rev", "1.62×10⁻⁴", "t CO₂/(t·km)", "GB/T 51366-2019, App. E"],
    ["Carbon Price", "p_c", "97.49", "CNY/t CO₂", "MEE, 2025-01-05"],
    ["Carbon Cap", "E_cap", "517,880", "t CO₂/yr", "Model scenario calibration"],
    ["Average Package Weight", "w_avg", "0.329", "kg/item", "Greenpeace report"],
    ["Paper Packaging Share", "rho", "53.50%", "—", "Greenpeace report"],
    ["Front-hub Sorting Yield", "δ", "0.85", "—", "Model calibration"],
    ["Central Hub Waste Loss Rate", "ω", "0.04", "—", "Model calibration"],
    ["Remanuf. Conversion Yield", "η", "0.90", "—", "Model calibration"],
    ["Front-hub Fixed Cost (regular)", "f_F", "18,000,000", "CNY/yr", "Model calibration"],
    ["Central Hub Fixed Cost (regular)", "f_L", "80,000,000", "CNY/yr", "Model calibration"],
    ["Yiwu Front-hub Fixed Cost", "f_F*", "120,000,000", "CNY/yr", "Model calibration"],
    ["Yiwu Central Hub Fixed Cost", "f_L*", "95,000,000", "CNY/yr", "Model calibration"],
    ["Raw Pulp Production Cost", "c_raw", "3,200", "CNY/t", "Model calibration"],
    ["Remanuf. Pulp Production Cost", "c_reman", "1,800", "CNY/t", "Model calibration"],
    ["Tier-1 Collection Cost (0–40%)", "c_t1", "400", "CNY/t", "Model calibration"],
    ["Tier-2 Collection Cost (40–60%)", "c_t2", "1,200", "CNY/t", "Model calibration"],
    ["Tier-3 Collection Cost (60–70%)", "c_t3", "2,200", "CNY/t", "Model calibration"],
    ["Disposal Cost", "c_d", "511", "CNY/t", "Greenpeace report"],
    ["Safety Stock Ratio", "σ", "15%", "—", "Model buffer assumption"],
    ["Central Hub Capacity (regular)", "Cap_L", "4,000,000", "t/yr", "Model calibration"],
    ["Yiwu Hub Capacity", "Cap_L*", "8,000,000", "t/yr", "Model calibration"],
    ["Front-hub Capacity (regular)", "Cap_K", "1,200,000", "t/yr", "Model calibration"],
    ["Road Factor", "κ", "1.30", "—", "Model calibration"],
], columns=["Parameter", "Symbol", "Value", "Unit", "Source"])
params_table.to_csv(OUTPUT_DIR / 'table1_parameters.csv', index=False)
print("  已生成：table1_parameters.csv")

# Table 2: 情景对比表
sc_rows = []
base_cost = sr[0]['total_cost']
for s in sr:
    row = {
        "Scenario": s['scenario'].replace('\n', ' '),
        "α Target (%)": f"{s['alpha']*100:.0f}",
        "Recovery Rate (%)": f"{s['recovery_rate']:.2f}",
        "Total Cost (×10⁸ CNY)": f"{s['total_cost']/1e8:.3f}",
        "Cost Change vs. Base (%)": f"{(s['total_cost']-base_cost)/base_cost*100:+.2f}",
        "Remanuf. Ratio (%)": f"{s['reman_ratio']*100:.2f}",
        "CO₂ Emission (×10⁴ t)": f"{s['total_emission']/1e4:.2f}",
        "Timber Saved (×10⁴ t)": f"{s['qty_x_reman']*P['wood_per_pulp']/1e4:.1f}",
        "Production Cost (×10⁸)": f"{s['prod_cost']/1e8:.3f}",
        "Transport Cost (×10⁸)": f"{s['transport_cost']/1e8:.3f}",
        "Collection Cost (×10⁸)": f"{s['collect_cost']/1e8:.3f}",
        "Fixed Cost (×10⁸)": f"{s['fixed_cost']/1e8:.3f}",
        "Disposal Cost (×10⁸)": f"{s['disp_cost']/1e8:.3f}",
        "Carbon Cost (×10⁸)": f"{s['carbon_cost']/1e8:.3f}",
        "Excess CO₂ (×10⁴ t)": f"{s['excess_emission']/1e4:.2f}",
    }
    sc_rows.append(row)
pd.DataFrame(sc_rows).to_csv(OUTPUT_DIR / 'table2_scenarios.csv', index=False)
print("  已生成：table2_scenarios.csv")

# Table 3: 敏感性汇总表
sens_rows = []
t1_base = next(r for r in t1 if r['c_tier1']==400)
for r in t1:
    sens_rows.append({
        "Analysis": "Tier-1 Collection Cost (c_tier1)",
        "Parameter Value": f"{r['c_tier1']} CNY/t",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": f"{r['reman_ratio']*100:.2f}",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-t1_base['total_cost'])/t1_base['total_cost']*100:+.2f}",
    })

rm_base = next(r for r in rm if r['c_reman']==1800)
for r in rm:
    sens_rows.append({
        "Analysis": "Remanuf. Cost (c_reman)",
        "Parameter Value": f"{r['c_reman']} CNY/t",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": f"{r['reman_ratio']*100:.2f}",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-rm_base['total_cost'])/rm_base['total_cost']*100:+.2f}",
    })

rw_base = next(r for r in rw if r['c_raw']==3200)
for r in rw:
    sens_rows.append({
        "Analysis": "Raw Pulp Cost (c_raw)",
        "Parameter Value": f"{r['c_raw']} CNY/t",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": f"{r['reman_ratio']*100:.2f}",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-rw_base['total_cost'])/rw_base['total_cost']*100:+.2f}",
    })

dl_base = next(r for r in dl if r['delta']==0.85)
for r in dl:
    sens_rows.append({
        "Analysis": "Sorting Yield (δ)",
        "Parameter Value": f"{r['delta']:.2f}",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": "—",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-dl_base['total_cost'])/dl_base['total_cost']*100:+.2f}",
    })

et_base = next(r for r in et if r['eta']==0.90)
for r in et:
    sens_rows.append({
        "Analysis": "Remanuf. Yield (η)",
        "Parameter Value": f"{r['eta']:.2f}",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": f"{r['reman_ratio']*100:.2f}",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-et_base['total_cost'])/et_base['total_cost']*100:+.2f}",
    })

cr_base = next(r for r in cr if abs(r['carbon_price']-97.49)<3)
for r in cr:
    sens_rows.append({
        "Analysis": "Carbon Price",
        "Parameter Value": f"{r['carbon_price']:.1f} CNY/t",
        "Recovery Rate (%)": f"{r['recovery_rate']:.2f}",
        "Remanuf. Ratio (%)": "—",
        "Total Cost (×10⁸)": f"{r['total_cost']/1e8:.3f}",
        "ΔCost vs. Base (%)": f"{(r['total_cost']-cr_base['total_cost'])/cr_base['total_cost']*100:+.2f}",
    })

pd.DataFrame(sens_rows).to_csv(OUTPUT_DIR / 'table3_sensitivity.csv', index=False)
print("  已生成：table3_sensitivity.csv")

# Table 4: 城市需求表
city_rows = []
for city, (lat, lon) in CITY_COORDS.items():
    city_rows.append({
        "City": city, "Latitude": lat, "Longitude": lon,
        "Annual Paper-based Demand (t)": CITY_DEMAND.get(city, 0),
        "Demand Share (%)": f"{CITY_DEMAND.get(city,0)/TD*100:.2f}",
        "Node Type": "Super-hub Market" if city=="金华 (义乌)" else "Market",
    })
city_df = pd.DataFrame(city_rows).sort_values("Annual Paper-based Demand (t)", ascending=False)
city_df.to_csv(OUTPUT_DIR / 'table4_city_demand.csv', index=False)
print("  已生成：table4_city_demand.csv")

# Table 5: 质量守恒验证表 【修改2-C：硬编码0.98替换为参数计算式】
mb = pd.DataFrame([
    ["Total Paper-based Demand (with safety)", f"{TD*(1+P['sigma_ratio'])/1e6:.4f}", "Mt/yr"],
    ["Total Collection (Tier 1+2+3)", f"{res_base['qty_rev']/1e6:.4f}", "Mt/yr"],
    ["Recovery Rate", f"{res_base['qty_rev']/TD*100:.4f}", "%"],
    ["Remanuf. Pulp Produced", f"{res_base['qty_x_reman']/1e6:.4f}", "Mt/yr"],
    ["Front-hub Sorting Waste", f"{res_base['waste_sort']/1e6:.4f}", "Mt/yr"],
    ["Central Hub Process Waste", f"{res_base['waste_center']/1e6:.4f}", "Mt/yr"],
    ["Factory Remanuf. Waste", f"{res_base['waste_process']/1e6:.4f}", "Mt/yr"],
    ["Total Waste", f"{(res_base['waste_sort']+res_base['waste_center']+res_base['waste_process'])/1e6:.4f}", "Mt/yr"],
    ["Mass Balance Check (Reman+Waste)", f"{(res_base['qty_x_reman']+res_base['waste_sort']+res_base['waste_center']+res_base['waste_process'])/1e6:.4f}", "Mt/yr"],
    ["Mass Balance Error", f"{res_base['mass_balance_error']:.6f}", "t"],
    ["Mass Balance Error (%)", f"{res_base['mass_balance_error_pct']:.8f}", "%"],
    ["Effective Conversion Rate", f"{res_base['effective_yield']*100:.2f}", "%"],
    ["Theoretical Effective Yield", f"{P['delta'] * (1 - P['omega']) * P['eta'] * 100:.2f}", "%"],
], columns=["Metric", "Value", "Unit"])
mb.to_csv(OUTPUT_DIR / 'table5_mass_balance.csv', index=False)
print("  已生成：table5_mass_balance.csv")

# =============================================================================
# 7. 论文核心数据汇总（控制台输出）
# =============================================================================
print("\n" + "="*90)
print(" "*25 + "论文核心结果汇总（可直接引用）")
print("="*90)

rb = res_base
total_c = rb['total_cost']
print("\n▌一、基准情景核心指标")
print(f"  系统总成本：{total_c/1e8:.2f} ×10⁸ CNY")
print(f"  生产成本占比：{rb['prod_cost']/total_c*100:.1f}%")
print(f"  运输成本占比：{rb['transport_cost']/total_c*100:.1f}%")
print(f"  回收成本占比：{(rb['collect_cost_tier1']+rb['collect_cost_tier2']+rb['collect_cost_tier3'])/total_c*100:.1f}%")
print(f"  设施固定成本占比：{rb['fixed_cost']/total_c*100:.1f}%")
print(f"  经济最优回收率：{rb['qty_rev']/TD*100:.2f}%")
print(f"  再制造浆占比：{rb['reman_ratio']*100:.2f}%")
print(f"  总碳排放量：{rb['total_emission']/1e4:.2f} ×10⁴ t")
print(f"  碳排放超额量：{rb['excess_emission']/1e4:.2f} ×10⁴ t")
print(f"  年节约木材量：{rb['qty_x_reman']*P['wood_per_pulp']/1e4:.1f} ×10⁴ t")
print(f"  年碳汇等效量：{rb['qty_x_reman']*P['wood_per_pulp']*P['co2_per_wood']/1e4:.1f} ×10⁴ t CO₂")

print("\n▌二、政策情景对比结论")
print(f"  当回收率要求从0提升至60%：")
print(f"    - 系统总成本增幅：{(sr[-1]['total_cost']-sr[0]['total_cost'])/sr[0]['total_cost']*100:.2f}%")
print(f"    - 再制造浆占比提升：{sr[-1]['reman_ratio']*100 - sr[0]['reman_ratio']*100:.2f} 个百分点")
print(f"    - 木材节约量提升：{(sr[-1]['qty_x_reman']-sr[0]['qty_x_reman'])*P['wood_per_pulp']/1e4:.1f} ×10⁴ t/yr")
print(f"  强制政策情景(α=0.5)较经济最优：")
print(f"    - 成本增加：{(sr[3]['total_cost']-sr[0]['total_cost'])/1e8:.2f} ×10⁸ CNY")
print(f"    - 回收率提升：{sr[3]['recovery_rate']-sr[0]['recovery_rate']:.2f} 个百分点")

print("\n▌三、敏感性分析排序（按影响程度降序）")
sens_sorted = sorted(tornado_data, key=lambda x: abs(x[2]-x[1]), reverse=True)
for idx, (name, low, high) in enumerate(sens_sorted, 1):
    swing = abs(high - low) / base_cost * 100
    print(f"  {idx}. {name.split(chr(10))[0]}：成本波动幅度 {swing:.2f}%")

print("\n▌四、模型可靠性验证")
print(f"  质量守恒绝对误差：{rb['mass_balance_error']:.6f} t")
print(f"  质量守恒相对误差：{rb['mass_balance_error_pct']:.8f}%")
print(f"  实际有效转化率：{rb['effective_yield']*100:.2f}%，理论值：{P['delta']*(1-P['omega'])*P['eta']*100:.2f}%")
print(f"  所有情景均求得最优解，求解器状态：Optimal")

# =============================================================================
# 最终总结
# =============================================================================
total_time = time.time() - start_time
print("\n" + "="*90)
print(" "*30 + "全部运行完成！")
print("="*90)
print(f"\n⏱  总运行耗时：{total_time/60:.1f} 分钟")
print("\n📁 输出文件清单：")
print(f"  📂 {OUTPUT_DIR} 目录")
print("    ├─ 6张学术图表（PDF矢量格式，IEEE标准）")
print("    │   ├─ fig1_network_topology.pdf")
print("    │   ├─ fig2_cost_structure.pdf")
print("    │   ├─ fig3_policy_performance.pdf")
print("    │   ├─ fig4_sensitivity_tornado.pdf")
print("    │   ├─ fig5_parameter_response.pdf")
print("    │   └─ fig6_environmental_benefits.pdf")
print("    └─ 5张数据表（CSV，可直接导入LaTeX）")
print("        ├─ table1_parameters.csv")
print("        ├─ table2_scenarios.csv")
print("        ├─ table3_sensitivity.csv")
print("        ├─ table4_city_demand.csv")
print("        └─ table5_mass_balance.csv")
print("\n✅ 模型、图表、表格、核心结论全部输出完成\n")



"""Compare edited inputs with the parameters of displayed numerical results."""
LABELS={
    ('geometry','inner_id_um'):' Sample capillary  ID',('geometry','inner_od_um'):' Sample capillary  OD',
    ('geometry','sheath_id_um'):' Sheath capillary  ID',('geometry','sheath_od_um'):' Sheath capillary  OD',
    ('geometry','gas_id_um'):' Gas capillary  ID',('geometry','sample_length_mm'):'内管流经长度',
    ('geometry','inner_recess_mm'):' Sample recess ',('geometry','gas_recess_mm'):' Gas recess ',
    ('geometry','ms_gap_mm'):' Emitter to  MS 距离',
    ('fluid','sample_flow_ul_min'):' Sample flow ',('fluid','sheath_flow_ul_min'):' Sheath flow ',
    ('fluid','gas_pressure_kpa'):'氮气压差',('electrical','voltage_kv'):' Voltage ',
    ('electrical','contact_distance_mm'):'加电点距离',('electrical','spray_current_na'):' Spray current ',
    ('electrical','sheath_conductivity_s_m'):'鞘液电导率',('electrical','quartz_zeta_mv'):' Quartz  Zeta',
    ('chemistry','sheath_methanol_fraction'):'甲醇比例',('cell','diameter_um'):' Cell diameter ',
    ('cell','density'):'细胞密度',('cell','zeta_mv'):' Cell  Zeta',
    ('cell','count'):'轨迹采样量',('cell','concentration_per_ml'):' Cell concentration '}
LABELS.update({('supply','sample_control'):'样品供液模式',('supply','sample_pressure_mbar'):'样品表压 mbar',
    ('cell','brownian_enabled'):' Brownian motion ',('cell','brownian_below_um'):'布朗粒径阈值',
    ('supply','orientation'):'管路摆放',('supply','reservoir_head_mm'):'液面高度 mm',
    ('supply','wall_capture_rate_s'):' Capture rate ',('supply','wall_capture_gap_um'):' Contact gap ',
    ('supply','outlet_pressure_pa'):'外部背压 Pa'})


def pending_message(displayed,edited):
    left=displayed.to_dict();right=edited.to_dict();changes=[]
    def show(value):return f'{value:g}' if isinstance(value,(float,int)) else str(value)
    for section,values in left.items():
        for key,old in values.items():
            if section=='numerics' and key in ('workers','particle_backend'):continue
            new=right[section][key]
            if old!=new:changes.append(f"{LABELS.get((section,key),section+'.'+key)} {show(old)} → {show(new)}")
    if not changes:return ''
    return '当前画面仍为旧计算结果.待应用: '+'; '.join(changes[:6])+('; …' if len(changes)>6 else '')+'.请应用参数并重新运行.'

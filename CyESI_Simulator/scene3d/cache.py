"""Reuse solved fields and responses for concentration-only continuous feeds."""
from copy import copy,deepcopy


def _signature(config):
    value=deepcopy(config.to_dict())
    value['cell'].pop('concentration_per_ml',None)
    if value['supply']['sample_control']=='pressure':value['fluid'].pop('sample_flow_ul_min',None)
    # The actual route is compared separately. The upstream integrator samples
    # it with more gravity segments than the hydraulic preview.
    value['supply'].pop('segments',None)
    return value


def reuse_concentration(scene,config,path,settings,phase_settings,phase_enabled,current_mode):
    from .version import solver_fingerprint
    if not hasattr(scene,'volume') or not hasattr(scene,'native_particles'):return None
    if scene.volume.diagnostics.get('model_fingerprint')!=solver_fingerprint():return None
    if _signature(scene.data.config)!=_signature(config):return None
    if scene.path is None or scene.path.to_dict()!=path.to_dict():return None
    if vars(scene.volume.settings)!=vars(settings):return None
    if scene.volume.diagnostics.get('current_mode','specified')!=current_mode:return None
    phase=getattr(scene.volume,'phase',None)
    if phase_enabled != (phase is not None):return None
    if phase is not None and vars(phase.settings)!=vars(phase_settings):return None
    if config.cell.concentration_per_ml==scene.data.config.cell.concentration_per_ml:return scene
    from .native import NativeExplorerData,NativeSceneData
    cfg=deepcopy(scene.data.config);cfg.cell.concentration_per_ml=config.cell.concentration_per_ml
    volume=copy(scene.volume);volume.diagnostics=deepcopy(scene.volume.diagnostics)
    volume.diagnostics['cache_reuse']='All physical fields, phase and finite-sphere responses reused; only the continuous source and acquisition ledger were rebuilt'
    data=NativeExplorerData(cfg,scene.data.fields,scene.data.summary,source=scene.data.source);data.volume=volume
    result=NativeSceneData(data,volume,scene.native_particles,scene,path)
    result.reference_source=scene.reference_source
    if hasattr(scene,'voltage_comparison'):result.voltage_comparison=scene.voltage_comparison
    return result


def reuse_fields(scene,config,path,settings,phase_settings,phase_enabled,current_mode):
    """Cell size/charge/response precision cannot change a dilute fluid solve."""
    if not hasattr(scene,'volume'):return None
    from .version import solver_fingerprint
    if scene.volume.diagnostics.get('model_fingerprint')!=solver_fingerprint():return None
    def key(c):
        value=_signature(c)
        value.pop('cell',None);value.pop('events',None);value.pop('spray',None)
        # In a dilute one-way model these change particle responses, not the
        # hydraulic/electric operating point or the background CFD solution.
        for name in ('wall_capture_rate_s','wall_capture_gap_um','saffman_enabled',
                     'lubrication_enabled','shear_damage_coefficient','shear_damage_exponent'):
            value['supply'].pop(name,None)
        value['numerics']={'flow_tolerance':c.numerics.flow_tolerance}
        if current_mode=='scaling':value['electrical'].pop('spray_current_na',None)
        return value
    if key(scene.data.config)!=key(config):return None
    if scene.path is None or scene.path.to_dict()!=path.to_dict():return None
    if vars(scene.volume.settings)!=vars(settings):return None
    if scene.volume.diagnostics.get('current_mode','specified')!=current_mode:return None
    phase=getattr(scene.volume,'phase',None)
    if phase_enabled!=(phase is not None):return None
    if phase is not None and vars(phase.settings)!=vars(phase_settings):return None
    volume=copy(scene.volume);volume.diagnostics=deepcopy(scene.volume.diagnostics)
    volume.diagnostics['cache_reuse']='Dilute gas/liquid/solvent/electric/phase fields reused; finite-sphere responses recomputed for changed cell parameters'
    return volume

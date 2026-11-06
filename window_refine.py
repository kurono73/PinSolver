"""Small, overlapping motion-only camera/zoom optimization windows."""
import numpy as np
from .reprojection import project_center_pose, robust_cost, robust_score, robust_weights


def _continuous_rvec(vector, previous):
    angle = np.linalg.norm(vector)
    if previous is None or angle < 1e-10:
        return vector
    axis = vector/angle
    return min((vector, vector-2*np.pi*axis, vector+2*np.pi*axis), key=lambda value:np.linalg.norm(value-previous))


def _refine_window(records, fixed_frames, strength, iterations):
    records = [dict(record) for record in records]
    origin = np.median(np.vstack([r['points'] for r in records]), axis=0)
    parameters = []
    mappings = []
    previous = None
    for record in records:
        record['points'] = record['points']-origin
        record['center'] = record['center']-origin
        record['rvec'] = _continuous_rvec(record['rvec'], previous)
        previous = record['rvec']
        mapping = {}
        if record['frame'] not in fixed_frames:
            mapping['rotation'] = slice(len(parameters),len(parameters)+3)
            parameters.extend(record['rvec'])
            if not record.get('fixed_center', False):
                mapping['center'] = slice(len(parameters),len(parameters)+3)
                parameters.extend(record['center'])
            if record.get('refine_focal', False):
                mapping['focal'] = len(parameters)
                parameters.append(np.log(record['intrinsics'][0,0]))
        mappings.append(mapping)
    if not parameters:
        return None
    parameters = np.array(parameters,dtype=np.float64)
    size = len(parameters)

    def state(values, index):
        record, mapping = records[index], mappings[index]
        rotation = values[mapping['rotation']] if 'rotation' in mapping else record['rvec']
        center = values[mapping['center']] if 'center' in mapping else record['center']
        matrix = record['intrinsics'].copy()
        if 'focal' in mapping:
            log_focal = values[mapping['focal']]
            if not -10 < log_focal < 20:
                return None
            ratio = matrix[1,1]/matrix[0,0]
            matrix[0,0] = np.exp(log_focal)
            matrix[1,1] = matrix[0,0]*ratio
        return rotation,center,matrix

    # Confidence uses subsequent observations, never the self-fitting birth frame.
    history = {}
    for i,record in enumerate(records):
        projected,_,_,_ = project_center_pose(record['points'],record['rvec'],record['center'],record['intrinsics'],record['distortion'])
        errors = np.linalg.norm(projected-record['pixels'],axis=1)
        for key,automatic,birth,error in zip(record['pin_ids'],record['automatic'],record['birth_frames'],errors):
            if automatic and record['frame'] != birth:
                history.setdefault(key,[]).append(error/record['scale'])
    for record in records:
        confidence = []
        for key,automatic in zip(record['pin_ids'],record['automatic']):
            samples = history.get(key,[])
            value = max(.25,1./max(1.,float(np.median(samples)))) if automatic and len(samples)>=2 else 1.
            confidence.append(value)
        weights = record['weights']*np.asarray(confidence)
        record['fit_weights'] = weights*record['weights'].sum()/max(weights.sum(),1e-12)

    def evaluate(values, with_jacobian=False):
        states = [state(values,i) for i in range(len(records))]
        if any(value is None for value in states):
            return None
        residuals, jacobians, data_scores = [], [], []
        cost = 0.
        scales = []
        for i,(record,mapping,current) in enumerate(zip(records,mappings,states)):
            rotation,center,matrix = current
            projected,pose_jac,intr_jac,depths = project_center_pose(record['points'],rotation,center,matrix,record['distortion'])
            if np.any(depths<=1e-8) or not np.isfinite(projected).all():
                return None
            errors = projected-record['pixels']
            weights = record['fit_weights']
            cost += robust_cost(errors,weights,record['scale'])
            data_scores.append(robust_score(errors,weights,record['scale']))
            row_weights = np.repeat(np.sqrt(weights*robust_weights(errors,record['scale'])),2)
            residuals.append(errors.reshape(-1)*row_weights)
            if with_jacobian:
                jacobian = np.zeros((errors.size,size))
                if 'rotation' in mapping:
                    jacobian[:,mapping['rotation']] = pose_jac[:,:3]
                if 'center' in mapping:
                    jacobian[:,mapping['center']] = pose_jac[:,3:6]
                if 'focal' in mapping:
                    jacobian[:,mapping['focal']] = intr_jac[:,6]*matrix[0,0]+intr_jac[:,7]*matrix[1,1]
                jacobians.append(jacobian*row_weights[:,None])
            focal_scale = matrix[0,0]*np.sqrt(weights.sum())
            scales.append((focal_scale/max(1e-8,float(np.median(depths))),focal_scale))

        def add_temporal(residual, terms):
            nonlocal cost
            residual = np.asarray(residual).reshape(-1)
            cost += .5*float(residual @ residual)
            residuals.append(residual)
            if with_jacobian:
                jacobian = np.zeros((len(residual),size))
                for column,coefficient in terms:
                    if isinstance(column,slice):
                        jacobian[:,column] += np.eye(len(residual))*coefficient
                    else:
                        jacobian[:,column] += coefficient
                jacobians.append(jacobian)

        for i in range(1,len(records)-1):
            dt0 = records[i]['frame']-records[i-1]['frame']
            dt1 = records[i+1]['frame']-records[i]['frame']
            time_scale = 2./(dt0+dt1)
            coefficients = (time_scale/dt0,-time_scale*(1./dt0+1./dt1),time_scale/dt1)
            neighborhood = range(i-1,i+2)
            condition = .5+.5*records[i].get('condition',0.)
            for field,slot,factor in (('center',1,.08*scales[i][0]),('rotation',0,.02*scales[i][1])):
                weight = strength*condition*factor
                acceleration = sum(coefficient*states[j][slot] for j,coefficient in zip(neighborhood,coefficients))
                terms = [(mappings[j][field],coefficient*weight) for j,coefficient in zip(neighborhood,coefficients) if field in mappings[j]]
                add_temporal(acceleration*weight,terms)
            if any('focal' in mappings[j] for j in neighborhood):
                weight = strength*.03*scales[i][1]
                acceleration = sum(coefficient*np.log(states[j][2][0,0]) for j,coefficient in zip(neighborhood,coefficients))
                terms = [(mappings[j]['focal'],coefficient*weight) for j,coefficient in zip(neighborhood,coefficients) if 'focal' in mappings[j]]
                add_temporal([acceleration*weight],terms)
        for i,record in enumerate(records):
            if record.get('guide') is not None and 'center' in mappings[i]:
                weight = record.get('guide_strength',0.)*.15*scales[i][0]
                add_temporal((states[i][1]-(record['guide']-origin))*weight,[(mappings[i]['center'],weight)])
        if not np.isfinite(cost):
            return None
        if with_jacobian:
            return cost,np.concatenate(residuals),np.vstack(jacobians),data_scores,states
        return cost,data_scores,states

    initial = evaluate(parameters)
    if initial is None:
        return None
    damping = .01
    for _ in range(iterations):
        current = evaluate(parameters,True)
        cost,residual,jacobian = current[:3]
        hessian = jacobian.T @ jacobian
        step = np.linalg.solve(hessian+damping*np.diag(np.maximum(np.diag(hessian),1e-10)),-jacobian.T @ residual)
        candidate = parameters+step
        trial = evaluate(candidate) if np.isfinite(candidate).all() else None
        if trial is not None and trial[0] < cost:
            parameters = candidate
            damping = max(1e-9,damping*.3)
        else:
            damping *= 10
        if np.linalg.norm(step)<1e-7 or damping>1e10:
            break
    final = evaluate(parameters)
    if final is None or final[0]>=initial[0]-1e-8:
        return None
    times = np.array([record['frame'] for record in records],dtype=float)
    old_steps = np.linalg.norm(np.diff([record['center'] for record in records],axis=0),axis=1)/np.diff(times)
    new_steps = np.linalg.norm(np.diff([value[1] for value in final[2]],axis=0),axis=1)/np.diff(times)
    if max(new_steps)>max(max(old_steps)*1.05,float(np.median(old_steps))*1.5)+1e-8:
        return None
    for i,record in enumerate(records):
        if final[1][i] > initial[1][i]+max(record['scale']*.12,initial[1][i]*.05):
            return None
        trusted = ~record['automatic']
        if trusted.sum()>=4:
            def trusted_score(value):
                projected,_,_,_ = project_center_pose(record['points'],value[0],value[1],value[2],record['distortion'])
                return robust_score((projected-record['pixels'])[trusted],record['weights'][trusted],record['scale'])
            before = project_center_pose(record['points'],record['rvec'],record['center'],record['intrinsics'],record['distortion'])[0]
            old = robust_score((before-record['pixels'])[trusted],record['weights'][trusted],record['scale'])
            if trusted_score(final[2][i]) > old+max(record['scale']*.12,old*.05):
                return None
    return [(rotation,center+origin,matrix) for rotation,center,matrix in final[2]]


def refine_camera_path(records, fixed_frames, mode, strength=1.):
    if len(records)<3 or mode=='OFF' or strength<=1e-6:
        return {},0
    records = [dict(record) for record in records]
    width = 15 if mode=='STRONG' else 7
    width = min(width,len(records))
    stride = max(1,width-2)
    starts = sorted(set(range(0,len(records)-width+1,stride))|{len(records)-width})
    changed = set()
    for start in starts:
        window = records[start:start+width]
        anchors = set(fixed_frames)|{window[0]['frame'],window[-1]['frame']}
        try:
            iterations = 4 if all(r.get('fixed_center',False) and not r.get('refine_focal',False) for r in window) else (8 if mode=='STRONG' else 6)
            result = _refine_window(window,anchors,strength*(1.6 if mode=='STRONG' else 1.),iterations)
        except (ValueError,np.linalg.LinAlgError,OverflowError):
            result = None
        if result is None:
            continue
        for index,(rotation,center,matrix) in enumerate(result,start):
            if records[index]['frame'] in anchors:
                continue
            records[index] = dict(records[index],rvec=rotation,center=center,intrinsics=matrix)
            changed.add(records[index]['frame'])
    return {record['frame']:record for record in records if record['frame'] in changed},len(changed)

"""As part of the ML for grain boundaries, we want to analyze the ML models that
we build to find the physics behind the predictions. This module includes
functions for calculating metrics from models related to feature importance.
"""
def order_features_by_gains(bst, feature_map_file):
    """Orders the features in the given XG booster by the information gain.

    Args:
        bst: booster from the XGBoost model (attribute `booster`).
        feature_map_file (str): path to the file that contains the feature map
          for the columns of the feature matrix.
    """
    str_dump = bst.get_dump(feature_map_file, with_stats=True)
    
    tree_arr = []
    for i_tree, tree in enumerate(str_dump):
        arr_lvls=tree.split('\n\t')
        a_tree = {}
        for lvl in arr_lvls:
            a_lvl ={}
            dum1 = lvl.split(',')
            if('leaf' in lvl):
                dum1[0].replace('\t','')
                dum10 = dum1[0].split(':')
                lvl_id = int(dum10[0])
                dum11 = dum10[1].split('leaf=')
                leaf = float(dum11[1])
                
                cover = float(dum1[1].replace('\n','').split('cover=')[1])
                a_lvl['lvl_id']=lvl_id
                a_lvl['leaf']=leaf
                a_lvl['cover']=cover
            else:
                dum10 = dum1[0].replace('\t','').replace('\n','')
                dum11 = dum10.split(':')
                lvl_id = int(dum11[0])
                dum12 = dum11[1].split('yes=')
                dum13 = dum12[0].replace('[','').replace(']','').split('<')
                feat_name = dum13[0]
                
                yes_to = int(dum12[1])
                no_to = int(dum1[1].split('no=')[1])
                missing = int(dum1[2].split('missing=')[1])
                gain = float(dum1[3].split('gain=')[1])
                cover = float(dum1[4].split('cover=')[1])            
                feat_thr = float(dum12[1])
                
                a_lvl['lvl_id']=lvl_id
                a_lvl['feat_name']=feat_name
                a_lvl['feat_thr'] = feat_thr
                a_lvl['yes_to'] = yes_to
                a_lvl['no_to']=no_to
                a_lvl['missing'] = missing
                a_lvl['gain']=gain
                a_lvl['cover']=cover
                
            a_tree[str(lvl_id)] = a_lvl
        tree_arr.append(a_tree)    
    feat_vocabulary = {}
    for tree in tree_arr:
        for lvl in tree:
            if('gain' in tree[lvl]):
                feat_data = feat_vocabulary.setdefault(tree[lvl]['feat_name'],{'gain':tree[lvl]['gain'],'cover':tree[lvl]['cover']})
                if(cmp(feat_data,{'gain':tree[lvl]['gain'],'cover':tree[lvl]['cover']})<>0):
                    try:
                        feat_vocabulary[tree[lvl]['feat_name']]['gain'] += tree[lvl]['gain']                    
                        feat_vocabulary[tree[lvl]['feat_name']]['cover'] += tree[lvl]['cover']
                    except:
                        feat_vocabulary[tree[lvl]['feat_name']]['gain'] = tree[lvl]['gain']                    
                        feat_vocabulary[tree[lvl]['feat_name']]['cover'] = tree[lvl]['cover']          
    
    sorted_feats = sorted(feat_vocabulary.items(),key=lambda k:k[1]['gain'], reverse=True)
    return sorted_feats

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from analysis import select_random_subset,find_max_bin_1d




def _default_clf_factory_map() -> Dict[str, Callable[[], object]]:
    """
    Return a mapping from short classifier name to a factory that creates
    a fresh classifier instance when called.
    """
    return {
        "RFC": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        "LogReg": lambda: LogisticRegression(
            penalty='l2', solver='saga', max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "GNB": lambda: GaussianNB(),
        "SVC": lambda: SVC(kernel='linear', C=1.0, probability=False, class_weight='balanced', random_state=42)
    }

def run_event_classification(
    iDict: Dict[str, Any],
    bsnm_list: List[Any],
    event_list: Optional[List[str]] = None,
    cond_list: Optional[List[str]] = None,
    clf_type_str: str = "LogReg",
    clf_factory_map: Optional[Dict[str, Callable[[], object]]] = None,
    train_prop: float = 0.75,
    nprs: int = 100,
    train_number: Optional[int] = None,
    match_train_number: bool = True,
    binarize: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run classification over events/conditions and return results per condition.
    - iDict: dict of event -> list/array per mouse -> dict of cond -> 2D array (cells x events)
    - bsnm_list: list of mouse identifiers (order must match iDict indexing)
    - clf_type_str: one of keys in clf_factory_map, e.g. 'LogReg', 'RFC', 'GNB', 'SVC'
    - clf_factory_map: optional map from name->factory; a sensible default is provided
    Returns:
    - all_clf_dat: dict keyed by condition containing per-event diagonals and 'both' (f1 scores)
    """
    if event_list is None:
        event_list = ['ds', 'swr']
    if cond_list is None:
        cond_list = ['prepulse', 'pulse', 'postpulse']
    if clf_factory_map is None:
        clf_factory_map = _default_clf_factory_map()

    clf_type_str = str(clf_type_str)
    if clf_type_str not in clf_factory_map:
        raise ValueError(f"Classifier '{clf_type_str}' not recognized. Available: {list(clf_factory_map.keys())}")

    if verbose:
        print(f"Using classifier: {clf_type_str}")

    labels = np.arange(len(event_list))
    n_event_type = len(event_list)
    n_mice = len(bsnm_list)

    all_clf_dat = {}

    for cond_indx, cond in enumerate(cond_list):
        output_conf_norm = np.zeros((n_event_type, n_event_type, n_mice))
        output_f1_score = np.zeros((n_mice,), dtype=float)

        if verbose:
            print(f"\nCondition: {cond} -> output shapes {output_conf_norm.shape}, {output_f1_score.shape}")

        odata = {}

        for mindx, mouse in enumerate(bsnm_list):
            if verbose:
                print(f" Mouse {mindx+1}/{n_mice}: id={mouse}")

            # infer n_cells from the first event's data for this mouse/cond
            try:
                n_cells = iDict[event_list[0]][mindx][cond].shape[0]
            except Exception as e:
                raise RuntimeError(f"Missing data for mouse {mouse}, event {event_list[0]}, cond {cond}: {e}")

            if verbose:
                print("  n_cells =", n_cells)

            train_data = np.array([]).reshape(n_cells, 0)
            test_data = np.array([]).reshape(n_cells, 0)
            train_labels = np.array([], dtype=int)
            test_labels = np.array([], dtype=int)

            max_bin = find_max_bin_1d(iDict, mindx, event_list, cond=cond)
            num_train_inds = int(max_bin * train_prop)
            num_test_inds = int(max_bin - num_train_inds)

            if verbose:
                print(f"  There are {max_bin} bins. training bins={num_train_inds}. Testing bins={num_test_inds}")

            for ev_indx, event in enumerate(event_list):
                label = ev_indx
                if verbose:
                    print(f"   event {event} -> label {label}")

                idata = iDict[event][mindx][cond]
                n_cells_check, n_events = idata.shape
                if n_cells_check != n_cells:
                    raise RuntimeError(f"Inconsistent n_cells for mouse {mouse}, event {event}")

                train_inds = generate_train_inds(n_events,
                                                 train_prop=train_prop,
                                                 train_number=train_number,
                                                 nprs=nprs,
                                                 pprint=False)
                test_inds = generate_test_inds(n_events, train_inds)

                temp_train_labels = np.repeat(label, len(train_inds))
                temp_test_labels = np.repeat(label, len(test_inds))

                temp_train_data = idata[:, train_inds]
                temp_test_data = idata[:, test_inds]

                if binarize:
                    temp_train_data = temp_train_data.astype(bool)
                    temp_test_data = temp_test_data.astype(bool)

                if match_train_number:
                    if verbose:
                        print("    match train and test number")
                    temp_train_data = select_random_subset(temp_train_data, num_train_inds, axis=1)
                    temp_test_data = select_random_subset(temp_test_data, num_test_inds, axis=1)
                    temp_train_labels = np.repeat(label, num_train_inds)
                    temp_test_labels = np.repeat(label, num_test_inds)

                train_data = np.concatenate((train_data, temp_train_data), axis=1)
                test_data = np.concatenate((test_data, temp_test_data), axis=1)
                train_labels = np.concatenate((train_labels, temp_train_labels), axis=0)
                test_labels = np.concatenate((test_labels, temp_test_labels), axis=0)

                if verbose:
                    print()

            X_train = train_data.T
            y_train = train_labels
            X_test = test_data.T
            y_test = test_labels

            if verbose:
                print(f"  Training data, X = {X_train.shape}, y = {y_train.shape}")
                print(f"  Testing data,  X = {X_test.shape}, y = {y_test.shape}")

            # guard: need at least one sample and >0 features
            if X_train.shape[0] > 0 and X_train.shape[1] > 0 and X_test.shape[0] > 0:
                clf = clf_factory_map[clf_type_str]()  # create a fresh instance
                try:
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    f1 = metrics.f1_score(y_test, y_pred, average="macro")
                    confusion_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
                    output_conf_norm[:, :, mindx] = confusion_norm
                    output_f1_score[mindx] = f1
                    if verbose:
                        print(f"  model f1 = {f1:.3f}")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: classifier failed for mouse {mouse}: {e}")
                    output_conf_norm[:, :, mindx] = np.nan
                    output_f1_score[mindx] = np.nan
            else:
                if verbose:
                    print("  Skipping training/prediction due to insufficient data")
                output_conf_norm[:, :, mindx] = np.nan
                output_f1_score[mindx] = np.nan

            # store per-event diagonal and f1 per mouse
            for ev_indx, ev in enumerate(event_list):
                odata.setdefault(ev, np.zeros(n_mice))
                odata[ev][mindx] = output_conf_norm[ev_indx, ev_indx, mindx]

            odata.setdefault('both', np.zeros(n_mice))
            odata['both'][mindx] = output_f1_score[mindx]

            if verbose:
                print()

        all_clf_dat[cond] = odata

    return all_clf_dat
    
def generate_train_inds(nEvents,train_prop=0.8,train_number=None,nprs=100,pprint=True):

    np.random.seed(nprs)
    
    if train_number is not None:
        train_size = int(train_number)
    else:
        train_size = int(train_prop * nEvents)
        
    test_size = nEvents - train_size
    if pprint:
        print('Total events:', nEvents)
        print('Training size:',train_size)
        print('Test size:',    test_size)

    return np.sort(np.random.choice(np.arange(nEvents),train_size,replace=False)).tolist()
    
def generate_test_inds(nEvents,train_inds):
    
    s = set(train_inds)
    li1 = np.arange(nEvents)
    
    return [x for x in li1 if x not in s]

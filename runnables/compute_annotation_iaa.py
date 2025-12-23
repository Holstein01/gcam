import krippendorff
import numpy as np
import pandas as pd
from pathlib import Path
from nltk import agreement
from nltk.metrics.distance import masi_distance


def create_annot(an, guidelines_to_classes=False):
    """
    Create frozensets with the unique label
    or with both labels splitting on pipe.
    Unique label has to go in a list so that
    frozenset does not split it into characters.
    """
    if "," in str(an):
        an = frozenset([item.strip() for item in an.split(",")])
    else:
        # single label has to go in a list
        # need to cast or not depends on your data
        an = frozenset([str(an)])

    if guidelines_to_classes:
        an = from_guidelines_to_classes(an)

    return an


def parse_annotations(ann_a_df, ann_b_df, guidelines_to_classes=False):
    samples = ann_a_df.shape[0]

    annotations = []
    for sample_id in range(samples):
        ann_a = ['ann_a', sample_id, create_annot(ann_a_df.iloc[sample_id].Annotation, guidelines_to_classes=guidelines_to_classes)]
        ann_b = ['ann_b', sample_id, create_annot(ann_b_df.iloc[sample_id].Annotation, guidelines_to_classes=guidelines_to_classes)]
        annotations.append(ann_a)
        annotations.append(ann_b)

    return annotations


def from_guidelines_to_classes(ann_set):
    subj_guidelines = {'G1', 'G2', 'G3', 'G4', 'G5'}
    if subj_guidelines.intersection(ann_set):
        return frozenset({'SUBJ'})

    return frozenset({'OBJ'})


def compute_iaa(annotations):
    # based on https://stackoverflow.com/questions/45741934/
    masi_task = agreement.AnnotationTask(distance=masi_distance)
    masi_task.load_array(annotations)
    # print("C: {}\nI: {}\nK: {}".format(masi_task.C, masi_task.I, masi_task.K))
    print("Alpha: {}".format(masi_task.alpha()))


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.resolve().joinpath('annotations')

    # First batch GCAM
    gcam_first_path = base_path.joinpath('first', 'gcam')
    first_a_gcam = pd.read_csv(gcam_first_path.joinpath('GCAM1.csv'))
    first_b_gcam = pd.read_csv(gcam_first_path.joinpath('GCAM2.csv'))
    first_gcam_annotations = parse_annotations(ann_a_df=first_a_gcam, ann_b_df=first_b_gcam)
    first_gcam_class_annotations = parse_annotations(ann_a_df=first_a_gcam, ann_b_df=first_b_gcam, guidelines_to_classes=True)
    print('First batch GCAM G_x IAA:')
    compute_iaa(first_gcam_annotations)
    print('First batch GCAM C_x IAA:')
    compute_iaa(first_gcam_class_annotations)
    print('\n')

    # First batch SAM
    sam_first_path = base_path.joinpath('first', 'sam')
    first_a_sam = pd.read_csv(sam_first_path.joinpath('SAM1.csv'))
    first_b_sam = pd.read_csv(sam_first_path.joinpath('SAM2.csv'))
    first_sam_annotations = parse_annotations(ann_a_df=first_a_sam, ann_b_df=first_b_sam)
    print('First batch SAM C_x IAA:')
    compute_iaa(first_sam_annotations)
    print('\n')

    # Second batch GCAM
    gcam_second_path = base_path.joinpath('second', 'gcam')
    second_a_gcam = pd.read_csv(gcam_second_path.joinpath('GCAM1.csv'))
    second_b_gcam = pd.read_csv(gcam_second_path.joinpath('GCAM2.csv'))
    second_gcam_annotations = parse_annotations(ann_a_df=second_a_gcam, ann_b_df=second_b_gcam)
    second_gcam_class_annotations = parse_annotations(ann_a_df=second_a_gcam, ann_b_df=second_b_gcam,
                                                     guidelines_to_classes=True)
    print('Second batch GCAM G_x IAA:')
    compute_iaa(second_gcam_annotations)
    print('Second batch GCAM C_x IAA:')
    compute_iaa(second_gcam_class_annotations)
    print('\n')

    # Second batch SAM
    sam_second_path = base_path.joinpath('second', 'sam')
    second_a_sam = pd.read_csv(sam_second_path.joinpath('SAM1.csv'))
    second_b_sam = pd.read_csv(sam_second_path.joinpath('SAM2.csv'))
    second_sam_annotations = parse_annotations(ann_a_df=second_a_sam, ann_b_df=second_b_sam)
    print('Second batch SAM C_x IAA:')
    compute_iaa(second_sam_annotations)
    print('\n')

    # All GCAM
    a_gcam = pd.concat((first_a_gcam, second_a_gcam), axis=0)
    b_gcam = pd.concat((first_b_gcam, second_b_gcam), axis=0)
    gcam_annotations = parse_annotations(ann_a_df=a_gcam, ann_b_df=b_gcam)
    gcam_class_annotations = parse_annotations(ann_a_df=a_gcam, ann_b_df=b_gcam, guidelines_to_classes=True)
    print('All GCAM G_x IAA:')
    compute_iaa(gcam_annotations)
    print('All GCAM C_x IAA:')
    compute_iaa(gcam_class_annotations)
    print('\n')

    # All SAM
    a_sam = pd.concat((first_a_sam, second_a_sam), axis=0)
    b_sam = pd.concat((first_b_sam, second_b_sam), axis=0)
    sam_annotations = parse_annotations(ann_a_df=a_sam, ann_b_df=b_sam)
    print('All SAM C_x IAA:')
    compute_iaa(sam_annotations)
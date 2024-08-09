import jiwer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

smoothing = SmoothingFunction()

transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    Calculates the BLEU score between a reference and a candidate translation.

    Args:
        reference (list[str]): The reference translation.
        candidate (list[str]): The candidate translation.
        weights (list[float], optional): The weights for each n-gram precision score. Defaults to [0.25, 0.25, 0.25, 0.25].

    Returns:
        float: The BLEU score between the reference and candidate translations.
    """
    score = sentence_bleu([reference.split()], candidate.split(), weights=weights, smoothing_function=smoothing.method2)
    return score

def wer(reference, candidate, transforms=transforms):
    """
    Calculate the Word Error Rate (WER) between two sequences of words.

    Args:
        reference (List[str]): The reference sequence of words.
        candidate (List[str]): The candidate sequence of words.
        transforms (jiwer.Compose, optional): The sequence of transformations to apply to the reference and candidate sequences. Defaults to the default sequence of transformations defined in the module.

    Returns:
        float: The WER score between the reference and candidate sequences.

    """
    wer = jiwer.wer(
                reference,
                candidate,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
    
    return wer

def cer(reference, candidate, transforms=transforms):
    """
    Calculate the Char Error Rate (CER) between two sequences of words.

    Args:
        reference (List[str]): The reference sequence of words.
        candidate (List[str]): The candidate sequence of words.
        transforms (jiwer.Compose, optional): The sequence of transformations to apply to the reference and candidate sequences. Defaults to the default sequence of transformations defined in the module.

    Returns:
        float: The WER score between the reference and candidate sequences.

    """
    cer = jiwer.cer(
                reference,
                candidate,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
    
    return cer

def mer(reference, candidate, transforms=transforms):
    """
    Calculate the Match Error Rate (MER) between two sequences of words.

    Args:
        reference (List[str]): The reference sequence of words.
        candidate (List[str]): The candidate sequence of words.
        transforms (jiwer.Compose, optional): The sequence of transformations to apply to the reference and candidate sequences. Defaults to the default sequence of transformations defined in the module.

    Returns:
        float: The WER score between the reference and candidate sequences.

    """
    mer = jiwer.mer(
                reference,
                candidate,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
    
    return mer
"""Correct only the package digest prefix used by the first static guard."""
import review

if __name__ == '__main__':
    # The canonical RA9 digest is relative to the particlegan directory.
    review.PKG = review.PKG / 'particlegan'
    review.main()

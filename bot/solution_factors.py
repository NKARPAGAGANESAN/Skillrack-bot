import math
import sys


def main():
    # Read input (works for both Skillrack and local testing)
    data = sys.stdin.read().strip().split()

    if not data:
        try:
            data = input().strip().split()
        except:
            return

    if not data:
        return

    try:
        n = int(data[0])
    except:
        return

    factors = []

    # Efficient factor calculation
    limit = int(math.isqrt(n))
    for i in range(1, limit + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)

    factors.sort()

    # Take first 3 factors (or less if not available)
    result = factors[:3]

    # Output
    print(" ".join(map(str, result)))


if __name__ == "__main__":
    main()
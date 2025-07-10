import numpy as np

class MycelialRosetta:
    def __init__(self):
        # Initialize linguistic and signal parameters
        self.base_weight = 1.0
        self.string_layers = 3  # Number of 'strings' wrapping around spike data
        self.triangle_inserts = True  # Triangular inflection modeling
        self.triangle_weight = 0.6

    def enhanced_w_transform(self, signal):
        """
        Enhanced W-transform integrating string-wrapping and triangle insert modeling.
        Input: signal (1D array of electrical measurements)
        Output: fingerprint dict including new feature extensions
        """

        # Baseline Fourier/W-transform (placeholder)
        freqs = np.fft.fftfreq(len(signal))
        fft_vals = np.fft.fft(signal)
        dominant_freq = freqs[np.argmax(np.abs(fft_vals))]

        # Base fingerprint metrics
        total_energy = np.sum(np.abs(signal) ** 2)
        peak_magnitude = np.max(np.abs(signal))

        # === Novel Enhancement Layer: String Wrapping and Triangle Inserts ===
        pattern_layers = []
        for i in range(1, self.string_layers + 1):
            modulated_signal = np.abs(signal) ** (1 / i)  # Layer encoding
            if self.triangle_inserts:
                triangle_pattern = self._triangle_wave(len(signal), i)
                modulated_signal *= triangle_pattern  # Composite structure
            energy_layer = np.sum(modulated_signal)
            pattern_layers.append(energy_layer * self.base_weight / i)

        # Final fingerprint
        fingerprint = {
            'dominant_frequency': np.abs(dominant_freq),
            'total_energy': total_energy,
            'peak_magnitude': peak_magnitude,
            'layered_energy_signature': pattern_layers,
            'nested_complexity': np.mean(pattern_layers)
        }
        return fingerprint

    def _triangle_wave(self, length, frequency_factor):
        """
        Simulate triangle structure overlay to mimic complex encoding.
        This represents 'triangles in strings' as you suggested.
        """
        x = np.linspace(0, 2 * np.pi, length)
        return 2 * np.abs(2 * ((x * frequency_factor / np.pi) % 1) - 1) - 1

    def analyze_fingerprint(self, fingerprint):
        """
        Interpret the fingerprint to estimate linguistic or biological meaning.
        """
        score = fingerprint['nested_complexity']
        if score > 100:
            classification = 'MULTILAYERED COMMUNICATION'
            hypothesis = 'Recursive or symbolic encoding present'
        elif score > 50:
            classification = 'DEEP STRUCTURE SIGNAL'
            hypothesis = 'Likely extended biological response'
        else:
            classification = 'SIMPLE ENCODED SPIKE'
            hypothesis = 'Direct electrical expression'
        return {
            'classification': classification,
            'hypothesis': hypothesis,
            'complexity_score': score
        }


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    np.random.seed(42)
    sample_signal = np.random.normal(0, 1, 1000)  # Simulated fungal spike train

    analyzer = MycelialRosetta()
    fingerprint = analyzer.enhanced_w_transform(sample_signal)
    interpretation = analyzer.analyze_fingerprint(fingerprint)

    print("\n📡 ENHANCED FUNGAL SIGNAL ANALYSIS")
    print("="*60)
    print(f"Dominant Frequency: {fingerprint['dominant_frequency']:.3f} Hz")
    print(f"Total Energy:       {fingerprint['total_energy']:.4f}")
    print(f"Peak Magnitude:     {fingerprint['peak_magnitude']:.4f}")
    print(f"Layered Energy Sig: {fingerprint['layered_energy_signature']}")
    print(f"Nested Complexity:  {fingerprint['nested_complexity']:.3f}")
    print("\n🔍 Interpretation:")
    print(f"Classification:     {interpretation['classification']}")
    print(f"Hypothesis:         {interpretation['hypothesis']}")
    print(f"Complexity Score:   {interpretation['complexity_score']:.3f}")

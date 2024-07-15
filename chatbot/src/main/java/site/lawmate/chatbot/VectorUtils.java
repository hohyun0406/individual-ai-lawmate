package site.lawmate.chatbot;

import java.util.List;

public class VectorUtils {
    public static double cosineSimilarity(List<Double> vector1, List<Double> vector2) {
        if (vector1.size() != vector2.size()) {
            throw new IllegalArgumentException("Vectors must have the same dimensions");
        }

        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        for (int i = 0; i < vector1.size(); i++) {
            dotProduct += vector1.get(i) * vector2.get(i);
            magnitude1 += Math.pow(vector1.get(i), 2);
            magnitude2 += Math.pow(vector2.get(i), 2);
        }

        double cosine = dotProduct / (Math.sqrt(magnitude1) * Math.sqrt(magnitude2));
        return cosine;
    }
}

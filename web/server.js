const express = require("express");
const multer = require("multer");
const path = require("path");
const { execFile } = require("child_process");
const fs = require("fs");
const os = require("os");

const app = express();
const PORT = 3000;

// Multer: save uploaded files to temp dir
const upload = multer({ dest: os.tmpdir() });

// Class metadata
const CLASS_INFO = {
	cardboard: {
		emoji: "📦",
		name: "Cartón",
		color: "#A0522D",
		tip: "Aplánalo antes de reciclarlo para ahorrar espacio.",
	},
	glass: {
		emoji: "🍶",
		name: "Vidrio",
		color: "#4FC3F7",
		tip: "Enjuágalo y deposítalo en el contenedor verde.",
	},
	metal: {
		emoji: "🥫",
		name: "Metal",
		color: "#78909C",
		tip: "Latas y aluminio van al contenedor amarillo.",
	},
	paper: {
		emoji: "📄",
		name: "Papel",
		color: "#FFF176",
		tip: "No mezcles papel mojado o con grasa.",
	},
	plastic: {
		emoji: "🧴",
		name: "Plástico",
		color: "#EF5350",
		tip: "Revisa el número de reciclaje en la base.",
	},
	trash: {
		emoji: "🗑️",
		name: "Basura General",
		color: "#9E9E9E",
		tip: "Este residuo no es reciclable, va al contenedor gris.",
	},
};

// Path to inference script
const INFERENCE_SCRIPT = path.resolve(__dirname, "..", "src", "inference.py");
const PYTHON_CMD = "python";

// Predict endpoint
app.post("/api/predict", upload.single("image"), (req, res) => {
	if (!req.file) {
		return res.status(400).json({ error: "No image provided." });
	}

	const imagePath = req.file.path;

	// Call Python inference script
	execFile(
		PYTHON_CMD,
		[INFERENCE_SCRIPT, imagePath],
		{ timeout: 30000 },
		(err, stdout, stderr) => {
			// Cleanup temp file
			fs.unlink(imagePath, () => {});

			if (err) {
				console.error("Inference error:", err.message);
				console.error("stderr:", stderr);
				return res
					.status(500)
					.json({ error: "Error processing image." });
			}

			try {
				const predictions = JSON.parse(stdout.trim());

				// Build response with metadata
				const results = predictions.map((p) => ({
					class: p.class,
					probability: p.probability,
					...CLASS_INFO[p.class],
				}));

				res.json({
					success: true,
					prediction: results[0],
					allResults: results,
				});
			} catch (parseErr) {
				console.error("Parse error:", parseErr.message);
				console.error("stdout:", stdout);
				res.status(500).json({
					error: "Error parsing prediction results.",
				});
			}
		},
	);
});

// Serve static files
app.use(express.static(path.join(__dirname, "public")));

// Start server
app.listen(PORT, () => {
	console.log(`\n🚀 TrashAI Server running at http://localhost:${PORT}\n`);
});

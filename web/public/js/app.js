// ═══════════════════════════════════════════
// TrashAI — Frontend Application Logic
// ═══════════════════════════════════════════

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadSection = document.getElementById("uploadSection");
const previewSection = document.getElementById("previewSection");
const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const previewImage = document.getElementById("previewImage");

let selectedFile = null;
let pieChart = null;
let barChart = null;

// ── Supported Formats ──
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];
const ALLOWED_MIME = [
	"image/jpeg",
	"image/png",
	"image/webp",
	"image/bmp",
	"image/gif",
];

function isValidImage(file) {
	const ext = file.name.split(".").pop().toLowerCase();
	return ALLOWED_MIME.includes(file.type) || ALLOWED_EXTENSIONS.includes(ext);
}

// ── Alert Modal ──
function showAlert(message) {
	// Remove existing alert if any
	const existing = document.getElementById("customAlert");
	if (existing) existing.remove();

	const overlay = document.createElement("div");
	overlay.id = "customAlert";
	overlay.style.cssText =
		"position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:9999;backdrop-filter:blur(4px);animation:fadeIn 0.2s ease";
	overlay.innerHTML = `
        <div style="background:#1f2937;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:32px;max-width:400px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,0.5);animation:slideUp 0.3s ease">
            <div style="font-size:3rem;margin-bottom:12px">⚠️</div>
            <h3 style="color:#f3f4f6;margin-bottom:8px;font-size:1.2rem">Formato no soportado</h3>
            <p style="color:#9ca3af;margin-bottom:20px;font-size:0.95rem">${message}</p>
            <button onclick="document.getElementById('customAlert').remove()" style="background:linear-gradient(135deg,#10b981,#059669);color:white;border:none;padding:10px 28px;border-radius:10px;font-weight:600;cursor:pointer;font-family:inherit">Entendido</button>
        </div>
    `;
	document.body.appendChild(overlay);
	overlay.addEventListener("click", (e) => {
		if (e.target === overlay) overlay.remove();
	});
}

// ── Dropzone Events ──
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
	e.preventDefault();
	dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
	dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
	e.preventDefault();
	dropzone.classList.remove("dragover");
	const file = e.dataTransfer.files[0];
	if (!file) return;
	if (!isValidImage(file)) {
		showAlert(
			`El archivo "<strong>${file.name}</strong>" no es una imagen válida.<br>Formatos aceptados: <strong>JPG, PNG, WEBP, BMP, GIF</strong>`,
		);
		return;
	}
	handleFile(file);
});

fileInput.addEventListener("change", (e) => {
	const file = e.target.files[0];
	if (!file) return;
	if (!isValidImage(file)) {
		showAlert(
			`El archivo "<strong>${file.name}</strong>" no es una imagen válida.<br>Formatos aceptados: <strong>JPG, PNG, WEBP, BMP, GIF</strong>`,
		);
		fileInput.value = "";
		return;
	}
	handleFile(file);
});

// ── File Handling ──
function handleFile(file) {
	selectedFile = file;
	const reader = new FileReader();
	reader.onload = (e) => {
		previewImage.src = e.target.result;
		showSection("preview");
	};
	reader.readAsDataURL(file);
}

// ── Section Management ──
function showSection(section) {
	uploadSection.classList.add("hidden");
	previewSection.classList.add("hidden");
	loadingSection.classList.add("hidden");
	resultsSection.classList.add("hidden");

	switch (section) {
		case "upload":
			uploadSection.classList.remove("hidden");
			break;
		case "preview":
			previewSection.classList.remove("hidden");
			break;
		case "loading":
			loadingSection.classList.remove("hidden");
			break;
		case "results":
			resultsSection.classList.remove("hidden");
			break;
	}
}

// ── Classification ──
async function classify() {
	if (!selectedFile) return;
	showSection("loading");

	const formData = new FormData();
	formData.append("image", selectedFile);

	try {
		const res = await fetch("/api/predict", {
			method: "POST",
			body: formData,
		});
		const data = await res.json();

		if (data.success) {
			renderResults(data);
		} else {
			alert("Error: " + (data.error || "Unknown error"));
			showSection("preview");
		}
	} catch (err) {
		console.error(err);
		alert("Error de conexión con el servidor.");
		showSection("preview");
	}
}

// ── Render Results ──
function renderResults(data) {
	const { prediction, allResults } = data;

	// Hero
	document.getElementById("resultEmoji").textContent = prediction.emoji;
	document.getElementById("resultClass").textContent = prediction.name;
	document.getElementById("confidenceText").textContent =
		`${prediction.probability}% de confianza`;
	document.getElementById("tipText").textContent = prediction.tip;

	// Animate confidence bar
	setTimeout(() => {
		document.getElementById("confidenceFill").style.width =
			`${prediction.probability}%`;
	}, 100);

	// Pie Chart
	renderPieChart(allResults);

	// Bar Chart
	renderBarChart(allResults);

	// Table
	renderTable(allResults);

	showSection("results");
}

// ── Pie Chart ──
function renderPieChart(results) {
	const ctx = document.getElementById("pieChart").getContext("2d");
	if (pieChart) pieChart.destroy();

	pieChart = new Chart(ctx, {
		type: "doughnut",
		data: {
			labels: results.map((r) => `${r.emoji} ${r.name}`),
			datasets: [
				{
					data: results.map((r) => r.probability),
					backgroundColor: results.map((r) => r.color),
					borderColor: "transparent",
					borderWidth: 0,
					hoverOffset: 8,
				},
			],
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			cutout: "55%",
			plugins: {
				legend: {
					position: "bottom",
					labels: {
						color: "#9ca3af",
						font: { family: "Inter", size: 11 },
						padding: 12,
						usePointStyle: true,
						pointStyleWidth: 8,
					},
				},
				tooltip: {
					backgroundColor: "#1f2937",
					titleColor: "#f3f4f6",
					bodyColor: "#9ca3af",
					borderColor: "rgba(255,255,255,0.06)",
					borderWidth: 1,
					cornerRadius: 8,
					padding: 12,
					callbacks: {
						label: (ctx) => ` ${ctx.parsed.toFixed(1)}%`,
					},
				},
			},
			animation: {
				animateRotate: true,
				duration: 1200,
				easing: "easeOutQuart",
			},
		},
	});
}

// ── Bar Chart ──
function renderBarChart(results) {
	const ctx = document.getElementById("barChart").getContext("2d");
	if (barChart) barChart.destroy();

	barChart = new Chart(ctx, {
		type: "bar",
		data: {
			labels: results.map((r) => r.emoji),
			datasets: [
				{
					data: results.map((r) => r.probability),
					backgroundColor: results.map((r) => r.color + "99"),
					borderColor: results.map((r) => r.color),
					borderWidth: 1,
					borderRadius: 6,
					borderSkipped: false,
				},
			],
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			indexAxis: "y",
			plugins: {
				legend: { display: false },
				tooltip: {
					backgroundColor: "#1f2937",
					titleColor: "#f3f4f6",
					bodyColor: "#9ca3af",
					borderColor: "rgba(255,255,255,0.06)",
					borderWidth: 1,
					cornerRadius: 8,
					padding: 12,
					callbacks: {
						title: (items) => {
							const idx = items[0].dataIndex;
							return results[idx].name;
						},
						label: (ctx) => ` ${ctx.parsed.x.toFixed(1)}%`,
					},
				},
			},
			scales: {
				x: {
					max: 100,
					grid: { color: "rgba(255,255,255,0.04)" },
					ticks: {
						color: "#6b7280",
						font: { family: "Inter", size: 11 },
					},
				},
				y: {
					grid: { display: false },
					ticks: {
						color: "#9ca3af",
						font: { family: "Inter", size: 16 },
					},
				},
			},
			animation: { duration: 1000, easing: "easeOutQuart" },
		},
	});
}

// ── Results Table ──
function renderTable(results) {
	const container = document.getElementById("resultsTable");
	container.innerHTML = results
		.map(
			(r) => `
        <div class="table-row">
            <span class="table-emoji">${r.emoji}</span>
            <span class="table-name">${r.name}</span>
            <div class="table-bar-wrapper">
                <div class="table-bar-fill" style="width:${r.probability}%;background:${r.color}"></div>
            </div>
            <span class="table-percentage" style="color:${r.color}">${r.probability}%</span>
        </div>
    `,
		)
		.join("");
}

// ── Reset ──
function resetApp() {
	selectedFile = null;
	fileInput.value = "";
	previewImage.src = "#";
	document.getElementById("confidenceFill").style.width = "0%";
	if (pieChart) pieChart.destroy();
	if (barChart) barChart.destroy();
	showSection("upload");
}

import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { useNavigate } from "react-router-dom";


const ClientPage = ({ user }) => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState("");
    const navigate = useNavigate();
    const [summary, setSummary] = useState({ accuracy: null, loss: null, contribution_score: null });
    const [globalSummary, setGlobalSummary] = useState({
        global_accuracy: null,
        global_loss: null,
        global_last_trained: null
    });

    useEffect(() => {
        if (!user) {
            navigate("/");
            return;
        }
        fetchClientSummary();
        fetchGlobalSummary();
    }, [user, navigate]);

    const fetchGlobalSummary = async () => {
        try {
            const response = await axios.get("/admin/summary");
            setGlobalSummary({
                global_accuracy: response.data.accuracy,
                global_loss: response.data.loss,
                global_last_trained: response.data.last_trained
            });
        } catch (error) {
            console.error("Failed to load global summary:", error);
            setMessage("Failed to load global summary.");
        }
    };

    const fetchClientSummary = async () => {
        try {
            const response = await axios.get("/client/summary");
            setSummary({
                accuracy: response.data.accuracy,
                loss: response.data.loss,
                contribution_score: response.data.contribution_score
            });
        } catch (error) {
            console.error("Failed to fetch client summary:", error);
            setMessage("Failed to fetch client summary.");
        }
    };

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            setMessage("Please select a file before uploading.");
            return;
        }

        const formData = new FormData();
        formData.append("client_id", user?.name || "Unknown_Client");
        formData.append("file", file);

        try {
            const response = await axios.post("/client/add-dataset", formData);
            setMessage(`Dataset uploaded successfully.`);
        } catch (error) {
            console.error("Error uploading dataset:", error);
            setMessage(error.response?.data?.error || "An error occurred while uploading the dataset.");
        }
    };

    const handleDownloadModel = async () => {
        try {
            const response = await axios.get("/client/download-model", { responseType: "blob" });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", "global_model.pth");
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error("Failed to download global model:", error);
            setMessage("Failed to download the global model.");
        }
    };

    return (
        <div className="min-h-screen flex flex-col bg-white">
        
            <main className="flex flex-col items-center p-4 w-full max-w-screen-lg mx-auto">
                <h1 className="text-lg font-semibold mb-4 text-center">Welcome, {user?.name || "Client"}</h1>

                <div className="flex flex-col sm:flex-row gap-4 w-full justify-center mb-4">
                    <input
                        className="border border-gray-300 rounded px-3 py-2 text-sm"
                        type="file"
                        onChange={handleFileChange}
                    />
                    <button
                        className="bg-black text-white px-4 py-2 rounded text-sm"
                        onClick={handleUpload}
                    >
                        Add Dataset
                    </button>
                    <button
                        className="bg-black text-white px-4 py-2 rounded text-sm"
                        onClick={handleDownloadModel}
                    >
                        Download Global Model
                    </button>
                </div>

                {message && (
                    <p className="text-sm text-center text-red-600 mb-4">{message}</p>
                )}

                <div className="flex flex-col sm:flex-row gap-4 w-full">
                    <div className="flex flex-col w-full sm:w-1/2 border border-gray-200 shadow-md rounded-md p-4">
                        <h2 className="text-md font-semibold mb-2 text-center">Client Summary</h2>
                        <p className="text-sm">Accuracy: {summary.accuracy ?? "N/A"}%</p>
                        <p className="text-sm">Loss: {summary.loss ?? "N/A"}%</p>
                        <p className="text-sm">Contribution Score: {summary.contribution_score ?? "N/A"}</p>
                    </div>

                    <div className="flex flex-col w-full sm:w-1/2 border border-gray-200 shadow-md rounded-md p-4">
                        <h2 className="text-md font-semibold mb-2 text-center">Global Summary</h2>
                        <p className="text-sm">Accuracy: {globalSummary.global_accuracy ?? "N/A"}%</p>
                        <p className="text-sm">Loss: {globalSummary.global_loss ?? "N/A"}%</p>
                        <p className="text-sm">Last Trained: {globalSummary.global_last_trained ?? "N/A"}</p>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default ClientPage;

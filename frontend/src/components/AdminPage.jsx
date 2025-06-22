import React, { useEffect, useState } from "react";
import axios from "../services/api";
import UpdateConfig from "./UpdateConfig";


const AdminPage = ({ user }) => {
    const [clients, setClients] = useState([]);
    const [error, setError] = useState("");
    const [isTraining, setIsTraining] = useState(false);
    const [summary, setSummary] = useState({
        accuracy: null,
        loss: null,
        last_trained: null
    });

    useEffect(() => {
        fetchClients();
        fetchSummary();
    }, []);

    const fetchSummary = async () => {
        try {
            const response = await axios.get("/admin/summary");
            setSummary({
                accuracy: response.data.accuracy,
                loss: response.data.loss,
                last_trained: response.data.last_trained
            });
        } catch (error) {
            setError("Failed to load global summary.");
        }
    };

    const fetchClients = async () => {
        try {
            const response = await axios.get("/admin/get-clients");
            setClients(response.data.clients);
        } catch (error) {
            setError("Failed to load client data.");
        }
    };

    const handleStartTraining = async () => {
        setIsTraining(true);
        try {
            const response = await axios.post("/admin/start-fl-training");
            alert(`Training completed. Accuracy: ${response.data.accuracy}, Loss: ${response.data.loss}`);
            await fetchSummary();
            await fetchClients();
        } catch (error) {
            alert("Failed to start training.");
        } finally {
            setIsTraining(false);
        }
    };

    return (
        <div className="min-h-screen bg-white flex flex-col">
          
            <main className="flex flex-col items-center px-4 py-6 w-full max-w-screen-xl mx-auto">
                <h1 className="text-2xl font-bold mb-4 text-center">Admin Page</h1>

                <button
                    className="bg-black text-white px-6 py-2 rounded text-sm mb-4"
                    onClick={handleStartTraining}
                    disabled={isTraining}
                >
                    {isTraining ? "Training..." : "Start Training"}
                </button>

                {error && <p className="text-sm text-red-600 mb-4">{error}</p>}

                <div className="flex flex-col lg:flex-row gap-6 w-full">
                    {/* Global Summary Card */}
                    <div className="flex-1 border border-gray-200 shadow-md rounded-md p-4">
                        <h2 className="text-md font-semibold mb-2 text-center">Global Summary</h2>
                        <p className="text-sm">Accuracy: {summary.accuracy ?? "N/A"}%</p>
                        <p className="text-sm">Loss: {summary.loss ?? "N/A"}</p>
                        <p className="text-sm">Last Trained: {summary.last_trained ?? "N/A"}</p>
                    </div>

                    {/* Client Table */}
                    <div className="flex-1 border border-gray-200 shadow-md rounded-md p-4 overflow-auto">
                        <h2 className="text-md font-semibold mb-2 text-center">Client Details</h2>
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-left border-b">
                                    <th className="p-2">Client ID</th>
                                    <th className="p-2">Dataset</th>
                                    <th className="p-2">Accuracy</th>
                                    <th className="p-2">Loss</th>
                                    <th className="p-2">Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                {clients.map((client) => (
                                    <tr key={client.id} className="border-b">
                                        <td className="p-2">{client.client_id}</td>
                                        <td className="p-2">{client.dataset_name}</td>
                                        <td className="p-2">{client.accuracy}</td>
                                        <td className="p-2">{client.loss}</td>
                                        <td className="p-2">{client.contribution_score}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Update Config */}
                    <div className="flex-1 border border-gray-200 shadow-md rounded-md p-4">
                        <h2 className="text-md font-semibold mb-2 text-center">Update Config</h2>
                        <UpdateConfig />
                    </div>
                </div>
            </main>
        </div>
    );
};

export default AdminPage;

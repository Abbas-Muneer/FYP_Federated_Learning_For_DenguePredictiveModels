import { Link } from "react-router-dom"


export default function Home(){
    return(
        <div className="h-screen flex flex-col items-center justify-center text-[#1b1b1b]">
            <div className="flex flex-col items-center text-center justify-center gap-5 p-20 ">
                <h6 className="text-4xl poppins-semibold">DengFL</h6>
                <p className="poppins-regular text-sm">A federated learning-based application for dengue prediction enables collaborative model training across distributed healthcare institutions without sharing raw patient data. It leverages local data from hospitals and clinics to build a global predictive model for early dengue detection. This approach ensures data privacy while improving the accuracy and scalability of dengue outbreak predictions.</p>
                <code className="font-bold text-l">Route through <Link className="cursor-pointer" to="/client"><span className="rounded-md font-bold p-1 bg-gray-100">/client</span></Link> for client terminal & <Link className="cursor-pointer" to="/admin"> <span className="rounded-md font-bold p-1 bg-gray-100">/admin</span> </Link> for admin dashboard</code>
            </div>
        </div>
    )
}
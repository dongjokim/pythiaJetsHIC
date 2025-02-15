#include "fastjet_compat.h"
#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TString.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace Pythia8;
using namespace fastjet;

// Function to display progress bar
void printProgress(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    clock_t cpu_start = clock();

    // Parse number of events from command line
    int nEvent = 100; // default value
    if (argc > 1) {
        nEvent = std::atoi(argv[1]);
    }
    std::cout << "Running with " << nEvent << " events" << std::endl;

    // Create ROOT histograms
    TH1F* hJetPt = new TH1F("hJetPt", "Jet p_{T};p_{T} (GeV/c);Counts", 100, 0, 100);
    TH1F* hJetEta = new TH1F("hJetEta", "Jet #eta;#eta;Counts", 100, -5, 5);

    // Initialize Pythia
    Pythia pythia;
    
    // Configure Pythia for pp collisions at 13 TeV with pthard bins
    pythia.readString("Beams:idA = 2212");     // proton
    pythia.readString("Beams:idB = 2212");     // proton
    pythia.readString("Beams:eCM = 13000.");   // 13 TeV

    // Turn on hard QCD processes
    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:inelastic = off");  // Turn off soft QCD

    // Set pthard range (in GeV)
    pythia.readString("PhaseSpace:pTHatMin = 20.");  // Minimum ptHat
    pythia.readString("PhaseSpace:pTHatMax = 100."); // Maximum ptHat

    pythia.readString("Print:quiet = on");        // Reduce Pythia output

    // Initialize Pythia
    if (!pythia.init()) {
        cout << "Error: Pythia initialization failed!" << endl;
        return 1;
    }

    // FastJet setup
    double R = 0.4;
    JetDefinition jet_def(antikt_algorithm, R);
    AreaDefinition area_def(active_area_explicit_ghosts);

    // Define structures for the TTree
    struct ParticleData {
        std::vector<float> pt;
        std::vector<float> eta;
        std::vector<float> phi;
        std::vector<int> jetIndex;
    };

    struct JetData {
        std::vector<float> pt;
        std::vector<float> eta;
        std::vector<float> phi;
        std::vector<float> area;
    };

    // Create output file
    TFile* outFile = new TFile("pythia_output.root", "RECREATE");

    // Create TTree and metadata
    TTree* tree = new TTree("events", "Event Data");

    // Create metadata variables (not vectors since they're constant for all events)
    TString jetAlgorithm = Form("Anti-kT, R=%.1f", jet_def.R());
    TString beamSpecies = "pp";  // or "PbPb", etc. - adjust as needed
    TString collisionEnergy = Form("#sqrt{s} = %.1f TeV", pythia.info.eCM()/1000.);

    // Add metadata branches
    tree->Branch("jetAlgorithm", &jetAlgorithm);
    tree->Branch("beamSpecies", &beamSpecies);
    tree->Branch("collisionEnergy", &collisionEnergy);

    // Declare the data structures
    ParticleData particles;
    JetData jets;

    // Create branches
    tree->Branch("particle_pt", &particles.pt);
    tree->Branch("particle_eta", &particles.eta);
    tree->Branch("particle_phi", &particles.phi);
    tree->Branch("particle_jetIndex", &particles.jetIndex);
    tree->Branch("jet_pt", &jets.pt);
    tree->Branch("jet_eta", &jets.eta);
    tree->Branch("jet_phi", &jets.phi);
    tree->Branch("jet_area", &jets.area);

    // Event loop
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
        if (!pythia.next()) continue;

        // Clear vectors for new event
        particles.pt.clear();
        particles.eta.clear();
        particles.phi.clear();
        particles.jetIndex.clear();
        jets.pt.clear();
        jets.eta.clear();
        jets.phi.clear();
        jets.area.clear();

        // Vector to store final-state particles for jet reconstruction
        std::vector<PseudoJet> fastjet_particles;  // Renamed from 'particles'

        // Loop over all particles in the event
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal() && pythia.event[i].isHadron()) {
                double eta = pythia.event[i].eta();
                double pt = pythia.event[i].pT();
                double phi = pythia.event[i].phi();
                
                // Wrap phi to [-π, π]
                while (phi > M_PI) phi -= 2*M_PI;
                while (phi < -M_PI) phi += 2*M_PI;
                
                // Store particle information for the tree
                particles.pt.push_back(pt);
                particles.eta.push_back(eta);
                particles.phi.push_back(phi);
                particles.jetIndex.push_back(-1);  // Will be updated after jet finding

                // Add to fastjet particles
                fastjet_particles.push_back(PseudoJet(
                    pythia.event[i].px(),
                    pythia.event[i].py(),
                    pythia.event[i].pz(),
                    pythia.event[i].e()
                ));
            }
        }

        // Perform jet clustering with area
        ClusterSequenceArea cs(fastjet_particles, jet_def, area_def);
        vector<PseudoJet> found_jets = sorted_by_pt(cs.inclusive_jets(5.0));

        // Store jet information and update particle-jet associations
        for (size_t i = 0; i < found_jets.size(); i++) {
            const PseudoJet& jet = found_jets[i];
            
            // Get jet properties and wrap phi
            double jet_phi = jet.phi();
            while (jet_phi > M_PI) jet_phi -= 2*M_PI;
            while (jet_phi < -M_PI) jet_phi += 2*M_PI;
            
            // Store jet properties
            jets.pt.push_back(jet.pt());
            jets.eta.push_back(jet.eta());
            jets.phi.push_back(jet_phi);  // Store wrapped phi
            jets.area.push_back(jet.area());

            // Update particle-jet associations
            vector<PseudoJet> constituents = jet.constituents();
            for (const PseudoJet& constituent : constituents) {
                for (size_t j = 0; j < fastjet_particles.size(); j++) {
                    if (constituent.user_index() == fastjet_particles[j].user_index()) {
                        particles.jetIndex[j] = i;
                    }
                }
            }
        }

        // Fill histograms
        for (const auto& jet : found_jets) {
            hJetPt->Fill(jet.pt());
            hJetEta->Fill(jet.eta());
        }

        // Fill the tree for this event
        tree->Fill();

        // Update progress bar
        printProgress(float(iEvent + 1) / nEvent);
    }
    std::cout << std::endl; // New line after progress bar

    // Save histograms
    hJetPt->Write();
    hJetEta->Write();

    // At the end of the program, before closing:
    tree->Write();  // Make sure to write the tree
    outFile->Write();
    outFile->Close();

    delete outFile;  // Clean up

    // Print final statistics
    pythia.stat();

    // Calculate and print timing information
    auto end_time = std::chrono::high_resolution_clock::now();
    clock_t cpu_end = clock();

    double cpu_time = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    auto wall_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "\nTiming Information:" << std::endl;
    std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
    std::cout << "Wall time: " << wall_time << " seconds" << std::endl;
    std::cout << "CPU/Wall: " << cpu_time/wall_time << std::endl;

    return 0;
}